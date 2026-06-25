"""Test mccl ACE collective operators"""

# pylint: disable=C0116,C0103
import pytest
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp

from torch_musa.testing.common_fsdp import FSDPTest
from torch_musa.testing.common_utils import get_musa_arch

NUM_DEVICES_FOR_TESTING = 4

skip = torch.musa.mccl.version() < (2, 28, 9) or get_musa_arch() < 31


@pytest.mark.skipif(skip, reason="Skip TestMCCLACECollectiveOps")
class TestMCCLACECollectiveOps(FSDPTest):
    """ "Test suit of mccl ACE collective operator"""

    @property
    def world_size(self) -> int:
        return NUM_DEVICES_FOR_TESTING

    @staticmethod
    def _allocate_symmetric_memory_pool(backend):
        pool = torch.musa.MemPool(backend.mem_allocator)
        backend.register_mem_pool(pool, symm=True)

        return pool

    def _get_collective_test_context(self):
        torch.musa.set_device(self.rank)
        device = torch.device(torch.musa.current_device())

        opts = dist.ProcessGroupMCCL.Options()
        # Enable Zero-CTA policy for CE collectives using config
        opts.config.cta_policy = dist.ProcessGroupMCCL.MCCL_CTA_POLICY_ZERO
        group = dist.new_group(
            backend="mccl", pg_options=opts, device_id=torch.device(self.rank)
        )

        backend = group._get_backend(device)
        return device, group, backend

    @staticmethod
    def _allocate_named_tensors(device, tensor_shapes):
        return {
            name: torch.empty(shape, dtype=torch.float32, device=device)
            for name, shape in tensor_shapes.items()
        }

    def _run_collective_parity_test(
        self,
        symm_tensor_shapes,
        golden_tensor_shapes,
        input_name,
        result_name,
        collective_fn,
    ):
        device, group, backend = self._get_collective_test_context()
        pool = self._allocate_symmetric_memory_pool(backend)

        with torch.musa.use_mem_pool(pool):
            symm_tensors = self._allocate_named_tensors(device, symm_tensor_shapes)

        golden_tensors = self._allocate_named_tensors(device, golden_tensor_shapes)
        symm_tensors[input_name].uniform_(-2, 2)
        golden_tensors[input_name].copy_(symm_tensors[input_name])

        collective_fn(group, symm_tensors)
        collective_fn(group, golden_tensors)

        torch.testing.assert_close(
            symm_tensors[result_name], golden_tensors[result_name]
        )

    def test_all_gather(self):
        inp_numel = 1024
        self._run_collective_parity_test(
            symm_tensor_shapes={
                "padding": (inp_numel,),
                "input": (inp_numel,),
                "result": (inp_numel * self.world_size,),
            },
            golden_tensor_shapes={
                "input": (inp_numel,),
                "result": (inp_numel * self.world_size,),
            },
            input_name="input",
            result_name="result",
            collective_fn=lambda group, tensors: dist.all_gather_into_tensor(
                tensors["result"], tensors["input"], group=group
            ),
        )

    def test_reduce_scatter(self):
        self.run_subtests(
            {
                "op": [
                    ReduceOp.SUM,
                ]
            },
            self._test_reduce_scatter,
        )

    def _test_reduce_scatter(self, op):
        output_numel = 1024
        self._run_collective_parity_test(
            symm_tensor_shapes={
                "padding": (output_numel * self.world_size,),
                "input": (output_numel * self.world_size,),
                "result": (output_numel,),
            },
            golden_tensor_shapes={
                "input": (output_numel * self.world_size,),
                "result": (output_numel,),
            },
            input_name="input",
            result_name="result",
            collective_fn=lambda group, tensors: dist.reduce_scatter_tensor(
                tensors["result"], tensors["input"], op=op, group=group
            ),
        )

    def test_all_reduce(self):
        self.run_subtests(
            {
                "op": [
                    ReduceOp.SUM,
                ]
            },
            self._test_all_reduce,
        )

    def _test_all_reduce(self, op):
        input_numel = 1024 * 512
        self._run_collective_parity_test(
            symm_tensor_shapes={
                # Reserve a prefix allocation so the tested tensors use a
                # non-zero offset within the symmetric-memory segment
                "offset_reservation": (input_numel),
                "input": (input_numel,),
            },
            golden_tensor_shapes={
                "input": (input_numel,),
            },
            input_name="input",
            result_name="input",
            collective_fn=lambda group, tensors: dist.all_reduce(
                tensors["input"], op=op, group=group
            ),
        )

    def test_all_to_all_single(self):
        # even split
        input_split_sizes = None
        output_split_sizes = None
        input_numel = output_numel = 4096
        meta_infos = [
            [input_split_sizes, output_split_sizes, input_numel, output_numel]
        ]

        input_split_sizes = [
            self.rank * self.world_size + peer_rank + 1
            for peer_rank in range(self.world_size)
        ]
        output_split_sizes = [
            src_rank * self.world_size + self.rank + 1
            for src_rank in range(self.world_size)
        ]
        input_numel = sum(input_split_sizes)
        output_numel = sum(output_split_sizes)
        meta_infos.append(
            [input_split_sizes, output_split_sizes, input_numel, output_numel]
        )
        self.run_subtests(
            {"input_meta_infos": meta_infos},
            self._test_all_to_all_single,
        )

    def _test_all_to_all_single(self, input_meta_infos):
        input_split_sizes, output_split_sizes, input_numel, output_numel = (
            input_meta_infos
        )

        self._run_collective_parity_test(
            symm_tensor_shapes={
                # Reserve a prefix allocation so the tested tensors use a
                # non-zero offset within the symmetric-memory segment
                "offset_reservation": (input_numel,),
                "input": (input_numel,),
                "result": (output_numel,),
            },
            golden_tensor_shapes={
                "input": (input_numel,),
                "result": (output_numel,),
            },
            input_name="input",
            result_name="result",
            collective_fn=lambda group, tensors: dist.all_to_all_single(
                tensors["result"],
                tensors["input"],
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            ),
        )
