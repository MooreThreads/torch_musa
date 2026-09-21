"""Test the functionality of Async TP fused operations"""

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_gather_tensor,
    reduce_scatter_tensor,
    wait_tensor,
)
from torch_musa.testing.common_dist import (
    MultiProcessingTest,
    skip_if_lt_x_gpu,
)

import torch_musa


NUM_DEVICES_FOR_TESTING_ASYNC_TP = 4


@skip_if_lt_x_gpu(NUM_DEVICES_FOR_TESTING_ASYNC_TP)
class TestAsyncTP(MultiProcessingTest):
    """Test the functionality of Async TP fused operations"""

    @property
    def world_size(self) -> int:
        return NUM_DEVICES_FOR_TESTING_ASYNC_TP

    def _get_default_group_name(self):
        group = dist.distributed_c10d._get_default_group()
        return dist._get_process_group_name(group)

    def _fused_mm_reduce_scatter(self, call_stack, tensor_a, tensor_b):
        group_name = self._get_default_group_name()
        if call_stack == "python":
            return dist._symmetric_memory._fused_matmul_reduce_scatter(
                tensor_a,
                tensor_b,
                reduce_op="sum",
                scatter_dim=0,
                group_name=group_name,
            )
        if call_stack == "op":
            return torch.ops.symm_mem.fused_matmul_reduce_scatter(
                tensor_a,
                tensor_b,
                "sum",
                0,
                group_name,
            )
        raise ValueError(f"Unknown call stack: {call_stack}")

    def _fused_all_gather_mm(self, call_stack, tensor_a, tensor_bs):
        group_name = self._get_default_group_name()
        if call_stack == "python":
            return dist._symmetric_memory._fused_all_gather_matmul(
                tensor_a,
                tensor_bs,
                gather_dim=0,
                group_name=group_name,
            )
        if call_stack == "op":
            return torch.ops.symm_mem.fused_all_gather_matmul(
                tensor_a,
                tensor_bs,
                0,
                group_name,
            )
        raise ValueError(f"Unknown call stack: {call_stack}")

    def test_fused_mm_reduce_scatter(self):
        """Test fused_matmul_reduce_scatter operation"""
        torch_musa.set_device(self.rank)
        device = torch.musa.current_device()

        for call_stack in ("python", "op"):
            with self.subTest(call_stack=call_stack):
                batch, m, n, k = 8, 64, 16, 32
                tensor_a = torch.randn(batch, m, k, device=device)
                tensor_b = torch.randn(k, n, device=device)

                result = self._fused_mm_reduce_scatter(call_stack, tensor_a, tensor_b)

                # Compute reference result
                matmul_result = tensor_a @ tensor_b  # shape: [batch, m, n]

                reference_result = reduce_scatter_tensor(
                    matmul_result, "sum", scatter_dim=0, group=self.process_group
                )
                reference_result = wait_tensor(reference_result)

                torch.testing.assert_close(result, reference_result)
                dist.barrier()

    def test_fused_all_gather_mm(self):
        """Test fused_all_gather_matmul operation"""
        torch_musa.set_device(self.rank)
        device = torch.musa.current_device()

        for call_stack in ("python", "op"):
            with self.subTest(call_stack=call_stack):
                batch, m, n, k = 8, 64, 16, 32
                tensor_a = torch.randn(batch, m, k, device=device)
                tensor_b = torch.randn(k, n, device=device)
                tensor_bs = [tensor_b.clone() for _ in range(8)]

                result = self._fused_all_gather_mm(call_stack, tensor_a, tensor_bs)

                # Compute reference result
                gathered_tensor_a = all_gather_tensor(
                    tensor_a, gather_dim=0, group=self.process_group
                )
                gathered_tensor_a = wait_tensor(gathered_tensor_a)
                reference_result = [
                    gathered_tensor_a @ tensor_b for tensor_b in tensor_bs
                ]

                torch.testing.assert_close(result[1], reference_result)
                dist.barrier()
