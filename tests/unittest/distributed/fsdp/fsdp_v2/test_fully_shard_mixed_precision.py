"""Test mixed precision training of FSDP2"""

# pylint: disable=invalid-name,missing-function-docstring

import copy
import functools
from typing import (
    Optional,
    Union,
)

import torch
from torch import nn
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _get_gradient_divide_factors,
)
from torch.testing._internal.common_fsdp import (
    MLP,
    patch_reduce_scatter,
    reduce_scatter_with_assert,
)

from torch_musa.testing.common_fsdp import (
    FSDPTest,
    skip_if_lt_x_gpu,
    check_sharded_parity,
)


NUM_GPUS_FOR_TESTING_FSDP_MP = 2


@skip_if_lt_x_gpu(NUM_GPUS_FOR_TESTING_FSDP_MP)
class TestFullyShardMixedPrecisionTraining(FSDPTest):
    """Test class for testing the correctness of FSDP2' mixed precision training"""

    @property
    def world_size(self) -> int:
        return NUM_GPUS_FOR_TESTING_FSDP_MP

    def _setup_models_and_optims(
        self,
        reshard_after_forward: Union[bool, int],
        param_dtype: Optional[torch.dtype],
        reduce_dtype: Optional[torch.dtype],
    ):
        torch.manual_seed(42)
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model).musa()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )
        for m in model:
            fsdp_fn(m)
        fsdp_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        return ref_model, ref_optim, model, optim

    def test_compute_dtype(self):
        self.run_subtests(
            {
                "param_dtype": [torch.bfloat16, torch.float16],
                "reshard_after_forward": [False, True],
            },
            self._test_compute_dtype,
        )

    def _test_compute_dtype(
        self, param_dtype: torch.dtype, reshard_after_forward: Union[bool, int]
    ):
        ref_model, ref_optim, model, optim = self._setup_models_and_optims(
            reshard_after_forward, param_dtype, None
        )
        ref_model_param_casted = copy.deepcopy(ref_model).to(param_dtype)

        def assert_fn(output: torch.Tensor):
            assert output.dtype == param_dtype

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, dist.reduce_scatter_tensor, assert_fn
        )
        # No replica, set all_reduce_group to None,
        predivide_factor, postdivide_factor = _get_gradient_divide_factors(
            self.process_group, all_reduce_group=None, reduce_dtype=param_dtype
        )
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device="musa", dtype=param_dtype)
        for idx in range(6):
            optim.zero_grad(set_to_none=idx % 2 == 0)
            fsdp_loss = model(inp).sum()
            with patch_reduce_scatter(reduce_scatter):
                fsdp_loss.backward()
            optim.step()

            ref_optim.zero_grad(set_to_none=idx % 2 == 0)
            ref_loss = ref_model_param_casted(inp.to(param_dtype)).sum()
            ref_loss.backward()
            for param in ref_model_param_casted.parameters():
                if predivide_factor and predivide_factor > 1:
                    param.grad.div_(predivide_factor)
                elif predivide_factor is None:
                    param.grad.div_(self.world_size)
                output = torch.zeros_like(torch.chunk(param.grad, self.world_size)[0])
                dist.reduce_scatter_tensor(output, param.grad)
                dist.all_gather_into_tensor(param.grad, output)
                if postdivide_factor and postdivide_factor > 1:
                    param.grad.div_(postdivide_factor)

            for param_fp32, param_half in zip(
                ref_model.parameters(), ref_model_param_casted.parameters()
            ):
                param_fp32.grad = param_half.grad.to(param_fp32.dtype)
                param_half.grad = None
            # run fp32 optimizer step
            ref_optim.step()

            for param_fp32, param_half in zip(
                ref_model.parameters(), ref_model_param_casted.parameters()
            ):
                param_half.detach().copy_(param_fp32)

            assert torch.allclose(fsdp_loss, ref_loss)
            check_sharded_parity(self, ref_model, model)

    def test_reduce_dtype(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "reduce_dtype": [torch.float32, torch.bfloat16],
            },
            self._test_reduce_dtype,
        )

    def _test_reduce_dtype(
        self, reshard_after_forward: Union[bool, int], reduce_dtype: torch.dtype
    ):
        param_dtype = torch.bfloat16
        ref_model, ref_optim, model, optim = self._setup_models_and_optims(
            reshard_after_forward, param_dtype, reduce_dtype
        )
        ref_model_param_casted = copy.deepcopy(ref_model).to(param_dtype)

        def assert_fn(output: torch.Tensor):
            assert output.dtype == reduce_dtype

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, dist.reduce_scatter_tensor, assert_fn
        )

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device="musa", dtype=param_dtype)
        for idx in range(6):
            optim.zero_grad(set_to_none=idx % 2 == 0)
            fsdp_loss = model(inp).sum()
            with patch_reduce_scatter(reduce_scatter):
                fsdp_loss.backward()
            optim.step()

            ref_optim.zero_grad(set_to_none=idx % 2 == 0)
            ref_loss = ref_model_param_casted(inp.to(param_dtype)).sum()
            ref_loss.backward()
            for param in ref_model_param_casted.parameters():
                if reduce_dtype == torch.bfloat16:
                    param_grad = param.grad.to(reduce_dtype)
                    sharded_grad = funcol.reduce_scatter_tensor(
                        param_grad,
                        scatter_dim=0,
                        reduceOp="avg",
                        group=self.process_group,
                    )
                    param.grad = funcol.all_gather_tensor(
                        sharded_grad, gather_dim=0, group=self.process_group
                    ).to(param.dtype)
                else:
                    param.grad.data = param.grad.to(torch.float32)
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            for param_fp32, param_half in zip(
                ref_model.parameters(), ref_model_param_casted.parameters()
            ):
                param_fp32.grad = param_half.grad.to(param_fp32.dtype)
                param_half.grad = None
            # run fp32 optimizer step
            ref_optim.step()

            for param_fp32, param_half in zip(
                ref_model.parameters(), ref_model_param_casted.parameters()
            ):
                param_half.detach().copy_(param_fp32)

            assert torch.allclose(fsdp_loss, ref_loss)
            check_sharded_parity(self, ref_model, model)
