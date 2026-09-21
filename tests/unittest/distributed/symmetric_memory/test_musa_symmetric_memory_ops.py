"""Test MUSA implementations registered for torch.ops.symm_mem."""

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symmetric_memory
from torch.distributed._symmetric_memory import (
    empty,
    enable_symm_mem_for_group,
    rendezvous,
)

import torch_musa  # pylint: disable=unused-import
from torch_musa import testing
from torch_musa.testing.common_dist import MultiProcessingTest, skip_if_lt_x_gpu

NUM_DEVICES_FOR_TESTING_SYMM_MEM_OPS = 4


@skip_if_lt_x_gpu(NUM_DEVICES_FOR_TESTING_SYMM_MEM_OPS)
class TestMUSASymmetricMemoryOps(MultiProcessingTest):
    """Test the MUSA symm_mem backend ops."""

    @property
    def world_size(self) -> int:
        return NUM_DEVICES_FOR_TESTING_SYMM_MEM_OPS

    def _get_default_group_name(self):
        group = dist.distributed_c10d._get_default_group()
        return dist._get_process_group_name(group)

    def _init_symm_tensor(self, numel, dtype=torch.float32):
        torch.musa.set_device(self.rank)
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        enable_symm_mem_for_group(group_name)
        tensor = empty(numel, dtype=dtype, device=device)
        symm_mem = rendezvous(tensor, group_name)
        symm_mem.barrier()
        return tensor, group_name

    def _rank_input(self, tensor):
        values = torch.arange(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        return values.reshape(tensor.shape) + self.rank

    def _all_reduce_expected(self, input_tensor):
        base = torch.arange(
            input_tensor.numel(), dtype=input_tensor.dtype, device=input_tensor.device
        ).reshape(input_tensor.shape)
        rank_sum = self.world_size * (self.world_size - 1) // 2
        return base * self.world_size + rank_sum

    def _assert_all_reduce_result(self, input_tensor, result):
        torch.testing.assert_close(
            result,
            self._all_reduce_expected(input_tensor),
            rtol=1e-5,
            atol=1e-5,
        )

    def _sync_ranks(self):
        torch.musa.synchronize()
        dist.barrier()

    def _reduce_scatter_expected(self, input_tensor, split_last_dim):
        reduced = self._all_reduce_expected(input_tensor)
        if split_last_dim:
            shard_size = reduced.size(-1) // self.world_size
            return reduced[..., self.rank * shard_size : (self.rank + 1) * shard_size]
        shard_size = reduced.numel() // self.world_size
        return reduced.flatten()[self.rank * shard_size : (self.rank + 1) * shard_size]

    def test_one_shot_all_reduce_ops(self):
        """Test one-shot all-reduce symm_mem op variants."""
        symm_tensor, group_name = self._init_symm_tensor(64)
        symm_tensor.copy_(self._rank_input(symm_tensor))
        self._sync_ranks()

        result = torch.ops.symm_mem.one_shot_all_reduce(symm_tensor, "sum", group_name)
        self._assert_all_reduce_result(symm_tensor, result)

        symm_tensor.copy_(self._rank_input(symm_tensor))
        self._sync_ranks()
        out = torch.empty_like(symm_tensor)
        returned = torch.ops.symm_mem.one_shot_all_reduce_out(
            symm_tensor, "sum", group_name, out
        )
        assert returned.data_ptr() == out.data_ptr()
        self._assert_all_reduce_result(symm_tensor, out)

        local_input = self._rank_input(symm_tensor)
        self._sync_ranks()
        result = torch.ops.symm_mem.one_shot_all_reduce_copy(
            symm_tensor, local_input, "sum", group_name
        )
        self._assert_all_reduce_result(local_input, result)

        self._sync_ranks()
        out = torch.empty_like(local_input)
        returned = torch.ops.symm_mem.one_shot_all_reduce_copy_out(
            symm_tensor, local_input, "sum", group_name, out
        )
        assert returned.data_ptr() == out.data_ptr()
        self._assert_all_reduce_result(local_input, out)
        dist.barrier()

    def test_two_shot_all_reduce_ops(self):
        """Test two-shot all-reduce symm_mem op variants."""
        symm_tensor, group_name = self._init_symm_tensor(64)
        symm_tensor.copy_(self._rank_input(symm_tensor))
        self._sync_ranks()

        returned = torch.ops.symm_mem.two_shot_all_reduce_(
            symm_tensor, "sum", group_name
        )
        assert returned.data_ptr() == symm_tensor.data_ptr()
        self._assert_all_reduce_result(symm_tensor, symm_tensor)
        self._sync_ranks()

        symm_tensor.copy_(self._rank_input(symm_tensor))
        self._sync_ranks()
        out = torch.empty_like(symm_tensor)
        returned = torch.ops.symm_mem.two_shot_all_reduce_out(
            symm_tensor, "sum", group_name, out
        )
        assert returned.data_ptr() == out.data_ptr()
        self._assert_all_reduce_result(symm_tensor, out)
        dist.barrier()

    def test_reduce_scatter_out(self):
        """Test reduce-scatter out op with both shard layouts."""
        for split_last_dim in (False, True):
            symm_tensor, group_name = self._init_symm_tensor(128)
            symm_tensor.copy_(self._rank_input(symm_tensor))
            self._sync_ranks()

            if split_last_dim:
                input_tensor = symm_tensor.view(2, 64)
                out = torch.empty(
                    2,
                    64 // self.world_size,
                    dtype=symm_tensor.dtype,
                    device=symm_tensor.device,
                )
            else:
                input_tensor = symm_tensor
                out = torch.empty(
                    symm_tensor.numel() // self.world_size,
                    dtype=symm_tensor.dtype,
                    device=symm_tensor.device,
                )

            returned = torch.ops.symm_mem.reduce_scatter_out(
                input_tensor, group_name, split_last_dim, out
            )
            assert returned.data_ptr() == out.data_ptr()
            torch.testing.assert_close(
                out,
                self._reduce_scatter_expected(input_tensor, split_last_dim),
                rtol=1e-5,
                atol=1e-5,
            )
            dist.barrier()

    def test_symm_memory_offset(self):
        """Test reduce-scatter when the input handle has a non-zero offset."""
        torch.musa.set_device(self.rank)
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        enable_symm_mem_for_group(group_name)

        allocator = symmetric_memory.get_mempool_allocator(device)
        mempool = torch.musa.MemPool(allocator)

        numel, dtype = (1024,), torch.float32
        with torch.musa.use_mem_pool(mempool):
            offset_reservation = torch.empty(numel, dtype=dtype, device=device)
            target = torch.empty_like(offset_reservation)

        base_hdl = symmetric_memory.rendezvous(offset_reservation, group=group_name)
        target_hdl = symmetric_memory.rendezvous(target, group=group_name)
        assert base_hdl.offset == 0
        assert target_hdl.offset == offset_reservation.untyped_storage().nbytes()
        assert target_hdl.offset > 0
        target_hdl.barrier()

        target.copy_(self._rank_input(target))
        self._sync_ranks()

        out = torch.empty(
            target.numel() // self.world_size,
            dtype=target.dtype,
            device=target.device,
        )
        returned = torch.ops.symm_mem.reduce_scatter_out(target, group_name, False, out)
        assert returned.data_ptr() == out.data_ptr()
        torch.testing.assert_close(
            out,
            self._reduce_scatter_expected(target, split_last_dim=False),
            rtol=1e-5,
            atol=1e-5,
        )
        dist.barrier()

    def test_tensor_sliced_offset(self):
        """Test reduce-scatter when input is a contiguous non-zero-offset view."""
        torch.musa.set_device(self.rank)
        device = torch.musa.current_device()
        group_name = self._get_default_group_name()
        enable_symm_mem_for_group(group_name)

        base = empty((20, 64), dtype=torch.float32, device=device)
        input_tensor = base[10:20]
        input_hdl = rendezvous(input_tensor, group_name)
        assert input_tensor.is_contiguous()
        assert input_tensor.storage_offset() > 0
        input_hdl.barrier()

        input_tensor.copy_(self._rank_input(input_tensor))
        self._sync_ranks()

        out = torch.empty(
            input_tensor.size(0),
            input_tensor.size(1) // self.world_size,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        returned = torch.ops.symm_mem.reduce_scatter_out(
            input_tensor, group_name, True, out
        )
        assert returned.data_ptr() == out.data_ptr()
        torch.testing.assert_close(
            out,
            self._reduce_scatter_expected(input_tensor, split_last_dim=True),
            rtol=1e-5,
            atol=1e-5,
        )
        dist.barrier()

    def test_uint32_write_validation(self):
        """Test uint32 write op argument validation."""
        torch.musa.set_device(self.rank)
        device = torch.musa.current_device()
        tensor = torch.empty(4, dtype=torch.uint32, device=device)

        with pytest.raises(RuntimeError, match="count must be a positive integer"):
            torch.ops.symm_mem.memset32_(tensor, 0, 1, 0)
        with pytest.raises(RuntimeError, match="offset \\+ count"):
            torch.ops.symm_mem.memset32_(tensor, 3, 1, 2)
        with pytest.raises(RuntimeError, match="offset .* exceeded"):
            torch.ops.symm_mem.stream_write_value32_(tensor, tensor.numel(), 1)
        with pytest.raises(RuntimeError, match="flat, contiguous uint32 tensor"):
            torch.ops.symm_mem.memset32_(tensor[::2], 0, 1, 1)


@pytest.mark.skipif(torch.musa.device_count() < 1, reason="Need at least 1 MUSA device")
class TestMUSASymmetricMemoryLocalOps:
    """Test single-device symm_mem ops and validations."""

    def setup_method(self):
        torch.musa.set_device(0)

    def test_two_shot_reduce_op_check(self):
        """Test two-shot all-reduce reduce_op validation."""
        device = torch.musa.current_device()
        tensor = torch.empty(4, dtype=torch.float32, device=device)
        out = torch.empty_like(tensor)

        with pytest.raises(RuntimeError, match="only sum is supported"):
            torch.ops.symm_mem.two_shot_all_reduce_(tensor, "max", "unused")
        with pytest.raises(RuntimeError, match="only sum is supported"):
            torch.ops.symm_mem.two_shot_all_reduce_out(tensor, "max", "unused", out)

    @pytest.mark.skipif(
        testing.get_musa_arch() < 22,
        reason="bf16 is not supported on arch older than qy2",
    )
    @pytest.mark.parametrize("is_b_row_major", [True, False])
    def test_async_input_mm(self, is_b_row_major):
        """Test MUSA implementation registered for symm_mem._async_input_mm."""
        device = torch.musa.current_device()
        a = torch.randn(512, 32, dtype=torch.bfloat16, device=device)
        if is_b_row_major:
            b = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
        else:
            b = torch.randn(64, 32, dtype=torch.bfloat16, device=device).t()
        a_chunk_signals = torch.ones(4, dtype=torch.uint32, device=device)

        result = torch.ops.symm_mem._async_input_mm(a, b, a_chunk_signals, 0)
        expected = torch.mm(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_uint32_write_ops(self):
        """Test local uint32 write ops."""
        device = torch.musa.current_device()
        tensor = torch.empty(8, dtype=torch.uint32, device=device)

        returned = torch.ops.symm_mem.memset32_(tensor, 0, 0, tensor.numel())
        assert returned.data_ptr() == tensor.data_ptr()
        returned = torch.ops.symm_mem.memset32_(tensor, 2, 0xFFFFFFFF, 3)
        assert returned.data_ptr() == tensor.data_ptr()

        torch.musa.synchronize()
        assert tensor.cpu().tolist() == [
            0,
            0,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0,
            0,
            0,
        ]

        try:
            returned = torch.ops.symm_mem.stream_write_value32_(tensor, 6, 17)
        except RuntimeError as err:
            if "operation not supported" in str(err):
                return
        assert returned.data_ptr() == tensor.data_ptr()
        assert tensor.cpu().tolist()[6] == 17
