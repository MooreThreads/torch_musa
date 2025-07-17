"""Test the MUSA Pluggable Allocator"""

import os
import ctypes
import tempfile
from pathlib import Path
import pytest
import torch

from torch_musa.utils.musa_extension import MUSA_HOME

from torch_musa.testing.common_utils import spawn_isolated_test


# pylint: disable=C0115,C0116


@spawn_isolated_test
def test_musa_pluggable_allocator():
    allocator_source = """
    #include <sys/types.h>
    #include <musa_runtime_api.h>

    extern "C" {
        int musa_malloc_cnt = 0;
        int musa_free_cnt = 0;

        void* my_alloc(size_t size, int device, void* stream) {
        void* ptr;
        musaMalloc(&ptr, size);
        musa_malloc_cnt += 1;  // single threaded scenario in our test case
        return ptr;
        }

        void my_free(void* ptr, size_t size, int device, void* stream) {
        musaFree(ptr);
        musa_free_cnt += 1;
        }
    }
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        src_file_path = Path(temp_dir) / "allocator.cpp"
        src_file_path.write_text(allocator_source)
        so_file_path = Path(temp_dir) / "allocator.so"

        compile_cmd = (
            f"g++ {src_file_path} -o {so_file_path} -I{MUSA_HOME}/include "
            f"-lmusart -L{MUSA_HOME}/lib -shared -fPIC"
        )
        os.system(compile_cmd)

        custom_allocator = torch.musa.memory.MUSAPluggableAllocator(
            so_file_path, "my_alloc", "my_free"
        )
        torch.musa.memory.change_current_allocator(custom_allocator)

        custom_alloc_lib = ctypes.CDLL(so_file_path)
        musa_malloc_cnt = ctypes.c_int.in_dll(custom_alloc_lib, "musa_malloc_cnt")
        musa_free_cnt = ctypes.c_int.in_dll(custom_alloc_lib, "musa_free_cnt")
        # before allocate memory
        assert (musa_malloc_cnt.value == 0) and (musa_free_cnt.value == 0)

        # allocate memory now
        t_0 = torch.randn((4,), device="musa")
        t_1 = torch.randn((4,), device="musa")
        assert (musa_malloc_cnt.value == 2) and (musa_free_cnt.value == 0)

        # check the correctness of computation
        out_musa = t_0 + t_1
        out_cpu = t_0.cpu() + t_1.cpu()
        assert torch.allclose(out_musa.cpu(), out_cpu)

        # free memory
        del t_0
        assert musa_free_cnt.value == 1
        del t_1
        assert musa_free_cnt.value == 2

        # after allocator initialization, try to set another allocator is not allowed
        another_custom_allocator = torch.musa.memory.MUSAPluggableAllocator(
            so_file_path, "my_alloc", "my_free"
        )
        with pytest.raises(
            RuntimeError, match="Can't swap an already initialized allocator"
        ):
            torch.musa.memory.change_current_allocator(another_custom_allocator)
