"""Test the functionality of SymmetricMemory"""

# pylint: disable=C0103
import os
from unittest.mock import patch

import pytest

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
import torch.distributed._symmetric_memory as symmetric_memory
from torch.distributed._symmetric_memory import (
    empty,
    rendezvous,
)

from torch_musa.testing import get_musa_arch
from torch_musa.testing.common_dist import (
    MultiProcessingTest,
    skip_if_lt_x_gpu,
)

NUM_DEVICES_FOR_TESTING_SYMM_MEM = 4


@pytest.mark.skipif(
    get_musa_arch() < 31, reason="Skip TestMUSASymmetricMemory on arch<31"
)
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

    def test_low_contention_all_gather_v_single(self):
        """Test variable element counts, graph reuse, and inplace offsets."""
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        dtype = torch.float32

        def rank_payload(rank, numel, payload_device):
            return (
                torch.arange(numel, dtype=torch.int64, device=payload_device)
                .add_(rank * 37)
                .remainder_(251)
                .to(dtype)
            )

        recv_count_cases = (
            [257, 509, 1021, 2039],
            [2053, 1033, 521, 269],
        )
        input_offset = 73
        max_input_numel = max(max(counts) for counts in recv_count_cases)
        input_backing = empty(
            input_offset + max_input_numel, dtype=dtype, device=device
        )

        # Reuse the same graph with changing per-rank transfer sizes.
        for recv_counts in recv_count_cases:
            local_input = input_backing.narrow(0, input_offset, recv_counts[self.rank])
            local_input.copy_(rank_payload(self.rank, recv_counts[self.rank], device))
            output = torch.empty(sum(recv_counts), dtype=dtype, device=device)
            torch.ops.symm_mem.low_contention_all_gather_v_single(
                output, local_input, recv_counts, group_name
            )
            expected = torch.cat(
                [
                    rank_payload(rank, numel, device)
                    for rank, numel in enumerate(recv_counts)
                ]
            )
            self.assertEqual(output, expected)

        # Inplace input uses the rank-dependent prefix offset. Also cover an
        # output view with a non-zero offset in its symmetric allocation.
        recv_counts = [263, 521, 1031, 2053]
        numel_offsets = [0]
        for numel in recv_counts[:-1]:
            numel_offsets.append(numel_offsets[-1] + numel)
        output_offset = 97
        output_backing = empty(
            output_offset + sum(recv_counts),
            dtype=dtype,
            device=device,
        )
        output = output_backing.narrow(0, output_offset, sum(recv_counts))
        local_input = output.narrow(0, numel_offsets[self.rank], recv_counts[self.rank])
        local_input.copy_(rank_payload(self.rank, recv_counts[self.rank], device))
        torch.ops.symm_mem.low_contention_all_gather_v_single(
            output, local_input, recv_counts, group_name
        )
        expected = torch.cat(
            [
                rank_payload(rank, numel, device)
                for rank, numel in enumerate(recv_counts)
            ]
        )
        self.assertEqual(output, expected)

    def test_low_contention_reduce_scatter_modes(self):
        self.run_subtests(
            {
                "reduce_op": ["sum", "avg"],
                "mode": ["copy", "prezeroed"],
                "inplace": [False, True],
            },
            self._test_low_contention_reduce_scatter_modes,
        )

    def _test_low_contention_reduce_scatter_modes(self, reduce_op, mode, inplace):
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        output_numel = 4096
        input_numel = output_numel * self.world_size

        # The prezeroed mode requires the caller to zero the receive buffer.
        env = {
            "TORCH_MUSA_SYMM_MEM_RS_OUTPUT_PREZEROED": (
                "1" if mode == "prezeroed" else "0"
            )
        }
        with patch.dict(os.environ, env):
            for pull_mode in (True,) if inplace else (False, True):
                input_tensor = (
                    empty(input_numel, dtype=torch.float32, device=device)
                    if pull_mode
                    else torch.empty(input_numel, dtype=torch.float32, device=device)
                )
                input_tensor.copy_(
                    torch.arange(input_numel, device=device, dtype=torch.float32)
                    .remainder_(97)
                    .add_(self.rank)
                )
                ref_output = torch.empty(
                    output_numel, dtype=torch.float32, device=device
                )
                torch.distributed.reduce_scatter_tensor(
                    ref_output,
                    input_tensor,
                    op=ReduceOp.AVG if reduce_op == "avg" else ReduceOp.SUM,
                )

                if inplace:
                    output_tensor = input_tensor.narrow(
                        0, self.rank * output_numel, output_numel
                    )
                elif pull_mode:
                    output_tensor = torch.empty(
                        output_numel, dtype=torch.float32, device=device
                    )
                else:
                    output_tensor = empty(
                        output_numel, dtype=torch.float32, device=device
                    )

                if not inplace:
                    output_tensor.fill_(23.0)
                    if mode == "prezeroed":
                        output_tensor.zero_()

                torch.ops.symm_mem.low_contention_reduce_scatter(
                    output_tensor, input_tensor, reduce_op, group_name
                )
                self.assertEqual(output_tensor, ref_output)

                if mode == "prezeroed" and not inplace:
                    # A nonzero buffer makes an unexpected internal memset
                    # observable: the preexisting value must survive.
                    output_tensor.fill_(23.0)
                    torch.ops.symm_mem.low_contention_reduce_scatter(
                        output_tensor, input_tensor, reduce_op, group_name
                    )
                    expected = ref_output + 23.0 / (
                        self.world_size if reduce_op == "avg" else 1
                    )
                    self.assertEqual(output_tensor, expected)


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


@skip_if_lt_x_gpu(NUM_DEVICES_FOR_TESTING_SYMM_MEM)
class TestSymmetricMemoryOpsNonCurrentDevice(MultiProcessingTest):
    """Test low-contention collectives on a non-current device."""

    @property
    def world_size(self) -> int:
        return NUM_DEVICES_FOR_TESTING_SYMM_MEM // 2

    def _get_default_group_name(self):
        group = dist.distributed_c10d._get_default_group()
        return dist._get_process_group_name(group)

    def test_low_contention_all_gather_v_single_non_current_device(self):
        """Use two ranks with disjoint current and target device sets."""
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        non_current_device = self.rank + self.world_size
        recv_counts = [271, 523]
        dtype = torch.float32

        def rank_payload(rank, numel, payload_device):
            return (
                torch.arange(numel, dtype=torch.int64, device=payload_device)
                .add_(rank * 37)
                .remainder_(251)
                .to(dtype)
            )

        with torch.musa.device(non_current_device):
            input_backing = empty(
                max(recv_counts),
                dtype=dtype,
                device=non_current_device,
            )
            local_input = input_backing.narrow(0, 0, recv_counts[self.rank])
            local_input.copy_(
                rank_payload(
                    self.rank,
                    recv_counts[self.rank],
                    non_current_device,
                )
            )
            output = torch.empty(
                sum(recv_counts),
                dtype=dtype,
                device=non_current_device,
            )

        torch.ops.symm_mem.low_contention_all_gather_v_single(
            output, local_input, recv_counts, group_name
        )
        self.assertEqual(torch.musa.current_device(), device)
        with torch.musa.device(non_current_device):
            torch.musa.synchronize()
            expected = torch.cat(
                [
                    rank_payload(rank, numel, non_current_device)
                    for rank, numel in enumerate(recv_counts)
                ]
            )
            self.assertEqual(output, expected)
        self.assertEqual(torch.musa.current_device(), device)
        torch.musa.synchronize()
