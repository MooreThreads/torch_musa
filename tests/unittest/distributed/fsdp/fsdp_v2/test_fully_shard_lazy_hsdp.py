"""Test Lazy HSDP All-Reduce functionality for FSDP2"""

# pylint: disable=invalid-name,missing-function-docstring
import functools

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._composable import checkpoint
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.common_fsdp import MLP

from torch_musa.testing.common_fsdp import FSDPTest, skip_if_lt_x_gpu

NUM_DEVICES_FOR_LAZY_HSDP = 4


@skip_if_lt_x_gpu(NUM_DEVICES_FOR_LAZY_HSDP)
class TestFullyShardHSDPTraining(FSDPTest):
    """Test numerical consistency of Lazy HSDP training"""

    @property
    def world_size(self) -> int:
        return min(4, NUM_DEVICES_FOR_LAZY_HSDP)

    def test_hsdp_training_parity(self):
        """Test training parity between lazy and non-lazy HSDP"""
        mesh = self._device_mesh_plan(hybrid_shard=True)
        self.run_subtests(
            {
                "use_activation_checkpointing": [False, True],
            },
            functools.partial(self._test_hsdp_training_parity, mesh),
        )

    def _test_hsdp_training_parity(
        self,
        mesh: DeviceMesh,
        use_activation_checkpointing: bool,
    ):
        mlp_dim = 16
        torch.manual_seed(42)

        non_lazy_model = nn.Sequential(
            nn.LayerNorm(mlp_dim, bias=True),
            MLP(mlp_dim, dim_multiplier=3),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=3),
        )

        torch.manual_seed(42)
        lazy_model = nn.Sequential(
            nn.LayerNorm(mlp_dim, bias=True),
            MLP(mlp_dim, dim_multiplier=3),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=3),
        )
        non_lazy_optim = torch.optim.Adam(non_lazy_model.parameters(), lr=1e-2)
        lazy_optim = torch.optim.Adam(lazy_model.parameters(), lr=1e-2)

        for mlp in non_lazy_model:
            if isinstance(mlp, MLP):
                if use_activation_checkpointing:
                    checkpoint(mlp)
                fully_shard(mlp, mesh=mesh)
        fully_shard(non_lazy_model, mesh=mesh)

        for mlp in lazy_model:
            if isinstance(mlp, MLP):
                if use_activation_checkpointing:
                    checkpoint(mlp)
                fully_shard(mlp, mesh=mesh)
        fully_shard(lazy_model, mesh=mesh)
        lazy_model.set_lazy_hsdp_allreduce(True, recurse=True)

        torch.manual_seed(42 + self.rank + 1)
        device = torch.musa.current_device()

        for iter_idx in range(5):
            num_microbatches = 3
            for microbatch_idx in range(num_microbatches):
                inp = torch.randn((8, mlp_dim), device=device)

                non_lazy_loss = non_lazy_model(inp).sum()
                non_lazy_loss.backward()

                lazy_loss = lazy_model(inp).sum()
                lazy_loss.backward()
                for (name, nl_param), l_param in zip(
                    non_lazy_model.named_parameters(),
                    lazy_model.parameters()
                ):
                    if nl_param.grad is None or l_param.grad is None:
                        raise AssertionError(
                            f"Parameter {name} grad is None: "
                            f"nl_grad={nl_param.grad}, l_grad={l_param.grad}"
                        )
                    nl_grad = nl_param.grad.full_tensor()
                    l_grad = l_param.grad.full_tensor()
                    assert torch.allclose(nl_grad, l_grad, rtol=1e-1, atol=1e-1), \
                        f"Grad mismatch for {name}: nl={nl_param.grad}, l={l_param.grad}"

                assert torch.allclose(non_lazy_loss, lazy_loss, rtol=1e-1, atol=1e-1), (
                    f"Iter {iter_idx}, microbatch {microbatch_idx}: "
                    f"loss mismatch - non_lazy: {non_lazy_loss}, lazy: {lazy_loss}"
                )
            non_lazy_optim.step()
            lazy_optim.step()

            non_lazy_optim.zero_grad(set_to_none=iter_idx % 2 == 0)
            lazy_optim.zero_grad(set_to_none=iter_idx % 2 == 0)
