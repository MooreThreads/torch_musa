"""Test GradScaler applied with FSDP model"""

# pylint: disable=invalid-name,missing-function-docstring
import copy

import torch
from torch import nn
from torch.distributed.fsdp import fully_shard

from torch_musa.core.amp.grad_scaler import GradScaler, OptState
from torch_musa.testing.common_fsdp import FSDPTest, skip_if_lt_x_gpu


NUM_GPUS_FOR_TESTING_FSDP_GRAD_SCALER = 2


@skip_if_lt_x_gpu(NUM_GPUS_FOR_TESTING_FSDP_GRAD_SCALER)
class TestFullyShardGradientScaler(FSDPTest):
    """Test class for testing the compatibility of GradScaler with FSDP"""

    @property
    def world_size(self) -> int:
        return NUM_GPUS_FOR_TESTING_FSDP_GRAD_SCALER

    def test_gradient_scaler(self):
        self.run_subtests(
            {"has_inf": [True, False]},
            self._test_gradient_scaler,
        )

    def _test_gradient_scaler(self, has_inf: bool):
        torch.manual_seed(42)
        model = nn.Sequential(
            # nn.Linear(4, 4, bias=False),
            *[nn.Linear(4, 4) for _ in range(2)]
        ).musa()
        for m in model:
            fully_shard(m)
        fully_shard(model)
        inp = torch.randn((4, 4), device="musa")

        scaler = GradScaler(init_scale=2.0, enabled=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss = model(inp).sum()
        scaler.scale(loss).backward()
        inv_scale = scaler._scale.reciprocal().float()
        if (
            has_inf
            and optimizer.param_groups[0]["params"][0].grad._local_tensor.device.index
            == 1
        ):
            optimizer.param_groups[0]["params"][0].grad._local_tensor[0, 0].fill_(
                float("inf")
            )
        inital_grad = optimizer.param_groups[0]["params"][0].grad.to_local().clone()

        scaler.unscale_(optimizer)
        for found_inf in scaler._per_optimizer_states[id(optimizer)][
            "found_inf_per_device"
        ].values():
            assert found_inf == has_inf
        assert (
            scaler._per_optimizer_states[id(optimizer)]["stage"].value
            == OptState.UNSCALED.value
        )

        unscaled_grad = optimizer.param_groups[0]["params"][0].grad.to_local().clone()
        assert torch.allclose(unscaled_grad, inital_grad * inv_scale)
        initial_scale = scaler.get_scale()
        initial_state = copy.copy(optimizer.state)

        scaler.step(optimizer)
        steped_state = copy.copy(optimizer.state)
        if has_inf:
            assert initial_state == steped_state
        else:
            assert initial_state != steped_state

        scaler.update()
        updated_scale = scaler.get_scale()
        if has_inf:
            backoff_factor = scaler.get_backoff_factor()
            assert updated_scale == initial_scale * backoff_factor
        else:
            assert updated_scale == initial_scale
