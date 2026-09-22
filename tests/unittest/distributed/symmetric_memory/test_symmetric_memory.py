"""Test the functionality of SymmetricMemory"""

# pylint: disable=C0103
import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
import torch.distributed._symmetric_memory as symmetric_memory
from torch.distributed._symmetric_memory import (
    empty,
    enable_symm_mem_for_group,
    rendezvous,
)

from torch_musa.testing.common_dist import (
    MultiProcessingTest,
    skip_if_lt_x_gpu,
)

NUM_DEVICES_FOR_TESTING_SYMM_MEM = 4


@skip_if_lt_x_gpu(NUM_DEVICES_FOR_TESTING_SYMM_MEM)
class TestMUSASymmetricMemory(MultiProcessingTest):
    """Test the functionality of the MUSA symmetric memory backend"""

    @property
    def world_size(self) -> int:
        return NUM_DEVICES_FOR_TESTING_SYMM_MEM

    def _get_default_group_name(self):
        # using WORLD group
        group = dist.distributed_c10d._get_default_group()
        return dist._get_process_group_name(group)

    def _init_symm_mem(self, size=128, dtype=torch.float32):
        torch.musa.set_device(self.rank)
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        enable_symm_mem_for_group(group_name)
        local_tensor = empty(size, dtype=dtype, device=device)
        symm_mem = rendezvous(local_tensor, group_name)
        return local_tensor, symm_mem

    def test_rendezvous(self):
        """test the allocation and rendezvous path of SymmetricMemory"""
        # pylint: disable=invalid-name
        local_tensor, symm_mem = self._init_symm_mem()

        # 1. check the memory allocation
        assert local_tensor.data_ptr() != 0, "memory allocation failed"

        # we can use this tensor locally
        t1 = local_tensor.zero_() + 1.0
        assert t1.all(), "all values of t1 should be 1.0"

        # 2. write(put) values into remote buffer then check the values locally
        for i in range(1, self.world_size):
            target_peer = (self.rank + i) % self.world_size
            target_buffer_tensor = symm_mem.get_buffer(
                target_peer, local_tensor.shape, local_tensor.dtype, 0
            )
            symm_mem.barrier()
            target_buffer_tensor.fill_(self.rank)
            symm_mem.barrier()
            src_peer = (self.rank + self.world_size - i) % self.world_size
            assert torch.all(local_tensor == src_peer), "got incorrect values"

    def test_put_signal(self):
        """test the put_signal path of MUSASymmetricMemory"""
        local_tensor, symm_mem = self._init_symm_mem()
        local_tensor.zero_()
        dist.barrier()

        if self.rank == 0:
            target_buffer_tensor = symm_mem.get_buffer(
                1, local_tensor.shape, local_tensor.dtype, 0
            )
            target_buffer_tensor.fill_(42.0)
            symm_mem.put_signal(dst_rank=1)
        elif self.rank == 1:
            symm_mem.wait_signal(src_rank=0)
            assert torch.all(local_tensor == 42.0), "rank 1 got incorrect values"

        dist.barrier()

    def test_wait_signal(self):
        """test the wait_signal path of MUSASymmetricMemory"""
        local_tensor, symm_mem = self._init_symm_mem()
        local_tensor.zero_()
        dist.barrier()

        if self.rank == 0:
            symm_mem.wait_signal(src_rank=1)
            assert torch.all(local_tensor == 24.0), "rank 0 got incorrect values"
        elif self.rank == 1:
            target_buffer_tensor = symm_mem.get_buffer(
                0, local_tensor.shape, local_tensor.dtype, 0
            )
            target_buffer_tensor.fill_(24.0)
            symm_mem.put_signal(dst_rank=0)

        dist.barrier()

    def test_barrier(self):
        """test the barrier path of MUSASymmetricMemory"""
        local_tensor, symm_mem = self._init_symm_mem()
        local_tensor.fill_(-1.0)

        target_peer = (self.rank + 1) % self.world_size
        target_buffer_tensor = symm_mem.get_buffer(
            target_peer, local_tensor.shape, local_tensor.dtype, 0
        )

        symm_mem.barrier()
        target_buffer_tensor.fill_(float(self.rank))
        symm_mem.barrier()

        src_peer = (self.rank + self.world_size - 1) % self.world_size
        assert torch.all(
            local_tensor == float(src_peer)
        ), "barrier got incorrect values"
        dist.barrier()


@skip_if_lt_x_gpu(NUM_DEVICES_FOR_TESTING_SYMM_MEM)
class TestSymmetricMemoryPool(MultiProcessingTest):
    """Test the functionality of the symmetric memory pool"""

    @property
    def world_size(self) -> int:
        return NUM_DEVICES_FOR_TESTING_SYMM_MEM

    def _get_default_group_name(self):
        # using WORLD group
        group = dist.distributed_c10d._get_default_group()
        return dist._get_process_group_name(group)

    def test_memory_pool(self):
        """Test memory pool by checking if handle offset is correctly set."""
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        enable_symm_mem_for_group(group_name)

        allocator = symmetric_memory.get_mempool_allocator(device)
        mempool = torch.musa.MemPool(allocator)

        numel, dtype = (1024,), torch.float32

        # allocate symmetric memory under mempool context
        with torch.musa.use_mem_pool(mempool):
            # x_0, x_1 share the same block
            x_0 = torch.empty(numel, dtype=dtype, device=device)
            x_1 = torch.empty_like(x_0)

        hdl0 = symmetric_memory.rendezvous(x_0, group=group_name)
        hdl1 = symmetric_memory.rendezvous(x_1, group=group_name)

        self.assertEqual(hdl0.offset, 0)
        self.assertEqual(hdl1.offset, x_0.untyped_storage().nbytes())


@skip_if_lt_x_gpu(NUM_DEVICES_FOR_TESTING_SYMM_MEM)
class TestSymmetricMemoryOps(MultiProcessingTest):
    """Test the functionality of the symmetric memory operators"""

    @property
    def world_size(self) -> int:
        return NUM_DEVICES_FOR_TESTING_SYMM_MEM

    def _get_default_group_name(self):
        # using WORLD group
        group = dist.distributed_c10d._get_default_group()
        return dist._get_process_group_name(group)

    def test_low_contention_all_gather(self):
        self.run_subtests(
            {
                "dtype": [torch.float32, torch.float16, torch.bfloat16],
            },
            self._test_low_contention_all_gather,
        )

    def _test_low_contention_all_gather(self, dtype):
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        enable_symm_mem_for_group(group_name)

        numel = 4096

        # input tensor in symmetric memory
        local_input_tensor_0 = torch.randn(numel, dtype=dtype, device=device)
        symm_mem_input_tensor = empty(numel, dtype=dtype, device=device)
        symm_mem_input_tensor.copy_(local_input_tensor_0)

        ref_output_tensor_0 = torch.empty(
            numel * self.world_size, dtype=dtype, device=device
        )
        torch.distributed.all_gather_into_tensor(
            ref_output_tensor_0, local_input_tensor_0, async_op=False
        )

        output_tensor_0 = symm_mem_input_tensor.new_empty(
            symm_mem_input_tensor.shape[0] * self.world_size,
            *symm_mem_input_tensor.shape[1:],
        )
        torch.ops.symm_mem.low_contention_all_gather(
            output_tensor_0, symm_mem_input_tensor, group_name
        )
        self.assertEqual(output_tensor_0, ref_output_tensor_0)

        output_tensor_1 = torch.ops.symm_mem._low_contention_all_gather(
            symm_mem_input_tensor, group_name
        )
        # _low_contention_all_gather uses backend_stream
        output_tensor_1 = torch.ops._c10d_functional.wait_tensor(output_tensor_1)
        self.assertEqual(output_tensor_1, ref_output_tensor_0)

        # inplace all-gather: input is the rank-local chunk of the output.
        inplace_output_tensor_0 = empty(
            numel * self.world_size, dtype=dtype, device=device
        )
        inplace_input_tensor_0 = inplace_output_tensor_0.chunk(self.world_size)[
            self.rank
        ]
        inplace_input_tensor_0.copy_(local_input_tensor_0)
        torch.ops.symm_mem.low_contention_all_gather(
            inplace_output_tensor_0, inplace_input_tensor_0, group_name
        )
        self.assertEqual(inplace_output_tensor_0, ref_output_tensor_0)

        # Also cover the same inplace pattern when output starts at a non-zero
        # offset inside a larger symmetric-memory allocation.
        offset_numel = 97
        backing_tensor = empty(
            offset_numel + numel * self.world_size, dtype=dtype, device=device
        )
        inplace_output_tensor_1 = backing_tensor.narrow(
            0, offset_numel, numel * self.world_size
        )
        inplace_input_tensor_1 = inplace_output_tensor_1.chunk(self.world_size)[
            self.rank
        ]
        inplace_input_tensor_1.copy_(local_input_tensor_0)
        torch.ops.symm_mem.low_contention_all_gather(
            inplace_output_tensor_1, inplace_input_tensor_1, group_name
        )
        self.assertEqual(inplace_output_tensor_1, ref_output_tensor_0)

    def test_low_contention_allreduce(self):
        self.run_subtests(
            {
                "dtype": [torch.float32, torch.float16, torch.bfloat16],
                "reduce_op": ["sum", "avg"],
            },
            self._test_low_contention_allreduce,
        )

    def _test_low_contention_allreduce(self, dtype, reduce_op):
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        enable_symm_mem_for_group(group_name)

        numel = 4096
        local_input = torch.randn(numel, dtype=dtype, device=device) + self.rank

        # Exercise a symmetric-memory view with a non-zero, aligned offset.
        offset_numel = 8
        backing = empty(numel + offset_numel, dtype=dtype, device=device)
        input_tensor = backing[offset_numel:]
        input_tensor.copy_(local_input)

        expected = local_input.clone()
        torch.distributed.all_reduce(
            expected,
            op=ReduceOp.AVG if reduce_op == "avg" else ReduceOp.SUM,
        )

        def assert_result(actual):
            if dtype == torch.float16:
                # CE atomics and MCCL may accumulate peers in a different order.
                self.assertEqual(actual, expected, rtol=2e-3, atol=1e-3)
            else:
                self.assertEqual(actual, expected)

        torch.ops.symm_mem.low_contention_allreduce(input_tensor, reduce_op, group_name)
        assert_result(input_tensor)

        # _low_contention_allreduce uses the input tensor directly when it is
        # allocated from symmetric memory.
        input_tensor.copy_(local_input)
        output = torch.ops.symm_mem._low_contention_allreduce(
            input_tensor, reduce_op, group_name
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert_result(output)
        assert_result(input_tensor)

        # A regular tensor is staged through the symmetric-memory workspace and
        # receives the reduced result in place.
        regular_input = local_input.clone()
        output = torch.ops.symm_mem._low_contention_allreduce(
            regular_input, reduce_op, group_name
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        assert_result(output)
        assert_result(regular_input)

    def test_low_contention_reduce_scatter(self):
        self.run_subtests(
            {
                "dtype": [torch.float32, torch.float16, torch.bfloat16],
                "reduce_op": ["sum", "avg"],
            },
            self._test_low_contention_reduce_scatter,
        )

    def _test_low_contention_reduce_scatter(self, dtype, reduce_op):
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        enable_symm_mem_for_group(group_name)

        output_numel = 4096
        input_numel = 4096 * self.world_size

        # push mode RS (recv buf on SymmetricMemory)
        input_tensor = torch.randn(input_numel, dtype=dtype, device=device)
        ref_output_tensor_0 = torch.empty(output_numel, dtype=dtype, device=device)
        torch.distributed.reduce_scatter_tensor(
            ref_output_tensor_0,
            input_tensor,
            op=ReduceOp.AVG if reduce_op == "avg" else ReduceOp.SUM,
            async_op=False,
        )
        output_tensor_0 = torch.ops.symm_mem._low_contention_reduce_scatter(
            input_tensor, reduce_op, group_name
        )

        # _low_contention_reduce_scatter uses backend_stream
        output_tensor_0 = torch.ops._c10d_functional.wait_tensor(output_tensor_0)
        self.assertEqual(output_tensor_0, ref_output_tensor_0)

        # invoke internal low_contention_reduce_scatter directly
        output_tensor_1 = empty(output_numel, dtype=dtype, device=device)
        torch.ops.symm_mem.low_contention_reduce_scatter(
            output_tensor_1, input_tensor, reduce_op, group_name
        )
        self.assertEqual(output_tensor_1, ref_output_tensor_0)

        # pull mode RS (send buf on SymmetricMemory)
        input_tensor_2 = empty(input_numel, dtype=dtype, device=device)
        input_tensor_2.copy_(input_tensor)
        output_tensor_2 = torch.ops.symm_mem._low_contention_reduce_scatter(
            input_tensor_2, reduce_op, group_name
        )
        output_tensor_2 = torch.ops._c10d_functional.wait_tensor(output_tensor_2)
        self.assertEqual(output_tensor_2, ref_output_tensor_0)


@pytest.mark.skipif(torch.musa.device_count() < 1, reason="requires a MUSA device")
def test_memory_pool_initialization_uses_current_stream():
    """Allocator initialization must not overwrite a non-default stream copy."""
    device = torch.device("musa", torch.musa.current_device())
    allocator = symmetric_memory.get_mempool_allocator(device.index)
    mempool = torch.musa.MemPool(allocator)
    copy_stream = torch.musa.Stream(priority=-1)

    # Cross the caching allocator's small/large segment boundary. Mimic FSDP
    # copy-in by writing one local shard of a larger communication buffer.
    for numel in (17, 1_024, 100_003, 1_000_000):
        with torch.musa.stream(copy_stream):
            source = torch.full((numel,), 7.0, dtype=torch.float32, device=device)
            with torch.musa.use_mem_pool(mempool):
                output = torch.empty(numel * 8, dtype=torch.float32, device=device)
            output[:numel].copy_(source)
            torch.musa.synchronize()
            assert torch.equal(output[:numel], source), f"copy corrupted: {numel=}"
