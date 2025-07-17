"""Test MemPool"""

import os
import ctypes

import threading
import tempfile
from pathlib import Path
import torch

from torch_musa.utils.musa_extension import MUSA_HOME
from torch_musa.testing.common_utils import spawn_isolated_test


# pylint: disable=C0115,C0116
def test_mempool_id():
    pool1 = torch.musa.graph_pool_handle()
    pool2 = torch.musa.MemPool().id

    assert pool1[0] == pool2[0]
    assert (pool2[1] - pool1[1]) > 0


def test_mempool_context():
    active_pool = torch.musa.MemPoolContext.active_pool()
    assert active_pool is None, "there should be no active pool"

    pool = torch.musa.MemPool()
    ctx = torch.musa.MemPoolContext(pool)
    active_pool = torch.musa.MemPoolContext.active_pool()
    assert active_pool is not None and (pool == active_pool)

    # leave out ctx, active pool is the previous one, i.e., None
    del ctx
    active_pool = torch.musa.MemPoolContext.active_pool()
    assert active_pool is None


def test_mempool_multithread():
    pool_ids, active_pool_ids = [], []

    def create_mempool_and_make_active():
        pool = torch.musa.MemPool()
        pool_ids.extend([pool.id])

        ctx = torch.musa.MemPoolContext(pool)
        active_pool = torch.musa.MemPoolContext.active_pool()
        active_pool_ids.extend([active_pool.id])
        del ctx

    num_threads = 4
    threads = [
        threading.Thread(target=create_mempool_and_make_active)
        for _ in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # each thread should create a unique mempool, since mempool id creation is atomic
    assert len(set(pool_ids)) == 4

    # each thread should have different active mempool, sice the pointer
    # to the active mempool is thread local
    assert len(set(active_pool_ids)) == 4


@spawn_isolated_test
def test_mempool_with_allocator():
    pool = torch.musa.MemPool()
    assert pool.allocator is None

    allocator_source = """
    #include <sys/types.h>
    #include <musa_runtime_api.h>

    extern "C" {
        int alloc_called_flag = 0;
        int free_called_flag = 0;

        void* my_alloc(size_t size, int device, void* stream) {
        void* ptr;
        musaMalloc(&ptr, size);
        alloc_called_flag = 1;
        return ptr;
        }

        void my_free(void* ptr, size_t size, int device, void* stream) {
        musaFree(ptr);
        free_called_flag = 1;
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
        pool = torch.musa.MemPool(custom_allocator.allocator())

        assert id(custom_allocator.allocator()) == id(pool.allocator)

        custom_alloc_lib = ctypes.CDLL(so_file_path)
        alloc_called_flag = ctypes.c_int.in_dll(custom_alloc_lib, "alloc_called_flag")
        free_called_flag = ctypes.c_int.in_dll(custom_alloc_lib, "free_called_flag")
        assert alloc_called_flag.value == 0
        assert free_called_flag.value == 0

        with torch.musa.use_mem_pool(pool):
            # will route to custom malloc logic
            _ = torch.randn((1,), device="musa")
            assert alloc_called_flag.value == 1
