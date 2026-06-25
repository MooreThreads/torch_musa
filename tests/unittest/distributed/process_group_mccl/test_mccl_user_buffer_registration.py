"""Test mccl user buffer registration"""

# pylint: disable=C0103, C0116
import os

import pytest
import torch
import torch.distributed as dist

from torch_musa.testing.common_fsdp import FSDPTest
from torch_musa.testing.common_utils import get_musa_arch


NUM_DEVICES_FOR_TESTING = 4

skip = torch.musa.mccl.version() < (2, 28, 9) or get_musa_arch() < 31


@pytest.mark.skipif(skip, reason="Skip TestMCCLUserBufferRegistration")
class TestMCCLUserBufferRegistration(FSDPTest):
    """ "Test suit of mccl user buffer registration"""

    def setUp(self) -> None:
        self._old_cta_policy = os.environ.get("MCCL_CTA_POLICY")
        if self._testMethodName == "test_mccl_window_registration":
            os.environ["MCCL_CTA_POLICY"] = "2"
        super().setUp()

    def tearDown(self) -> None:
        try:
            super().tearDown()
        finally:
            if self._testMethodName == "test_mccl_window_registration":
                if self._old_cta_policy is None:
                    os.environ.pop("MCCL_CTA_POLICY", None)
                else:
                    os.environ["MCCL_CTA_POLICY"] = self._old_cta_policy

    @property
    def world_size(self) -> int:
        return NUM_DEVICES_FOR_TESTING

    def test_mccl_user_buffer_registration(self):
        torch.musa.set_device(self.rank)
        device = torch.device(torch.musa.current_device())

        pg = dist.distributed_c10d._get_default_group()
        backend = pg._get_backend(device)
        # Use MCCL memory allocator
        # enable user buffer registration usage in MCCL
        pool = torch.musa.MemPool(backend.mem_allocator)

        # allocate memory with mcclMemAlloc
        with torch.musa.use_mem_pool(pool):
            inp = torch.arange(1024 * 1024 * 2, device=device)

        # register buffers to MCCL (previous allocated inp tensor should be registered)
        backend.register_mem_pool(pool)

        # use Switches ?
        dist.all_reduce(inp)
        torch.musa.synchronize()

        # de-register buffers from MCCL
        backend.deregister_mem_pool(pool)

        # clean up memory
        del inp, pool

    def test_mccl_window_registration(self):
        torch.musa.set_device(self.rank)
        device = torch.device(torch.musa.current_device())

        pg = dist.distributed_c10d._get_default_group()
        backend = pg._get_backend(device)

        # Use MCCL memory allocator
        # enable symmetric memory usage in MCCL
        pool = torch.musa.MemPool(backend.mem_allocator)

        # allocate memory with mcclMemAlloc
        with torch.musa.use_mem_pool(pool):
            inp = torch.arange(1024 * 1024 * 2, device=device, dtype=torch.float32)

            output = torch.arange(
                1024 * 1024 * 2 * self.world_size, device=device, dtype=torch.float32
            )

        # register buffers to MCCL (previous allocated inp tensor should be registered)
        backend.register_mem_pool(pool, symm=True)

        # allgather now should use Async Copy Engine
        dist.all_gather_into_tensor(output, inp)

        # further allocations are also registered
        with torch.musa.use_mem_pool(pool):
            inp = torch.arange(1024 * 1024 * 2, device=device, dtype=torch.float32)
        dist.all_gather_into_tensor(output, inp)

        # de-register buffers from MCCL
        backend.deregister_mem_pool(pool)

        # clean up memory
        del inp, output, pool
