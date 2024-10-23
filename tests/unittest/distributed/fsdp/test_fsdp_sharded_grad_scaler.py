"""test fsdp ShardGradScaler"""

# pylint: disable=invalid-name
import pytest

import torch
from torch_musa import testing
from torch_musa.distributed.fsdp import ShardedGradScaler
from torch_musa.testing.common_fsdp import FSDPTest, skip_if_lt_x_gpu


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="Skip on arch's version older than 22"
)
class TestShardGradScaler(FSDPTest):
    """class TestShardGradScaler"""

    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_grad_scaling(self):
        # NOTE: `test_xxx' does not support additional function arguments currently
        self.run_subtests(
            {
                "device": ["cpu", "musa"],  # FSDP maybe offload to cpu
            },
            self._test_grad_scaling,
        )

    def _test_grad_scaling(self, device):
        scaler = ShardedGradScaler(init_scale=2.0, enabled=True)
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device=device)
        t1 = torch.full((1,), 8.0, dtype=torch.float32, device=device)
        outputs = [t1.clone(), (t0.clone(), t1.clone()), [t0.clone(), t1.clone()]]
        outputs = scaler.scale(outputs)
        self.assertTrue(
            outputs[0] == 16.0 and outputs[1][0] == 8.0 and outputs[1][1] == 16.0
        )
        self.assertTrue(outputs[2][0] == 8.0 and outputs[2][1] == 16.0)
        self.assertTrue(scaler._scale.device == t1.device)

    @skip_if_lt_x_gpu(2)
    def test_inf_gradients_skip_optim_step(self):
        self.run_subtests(
            {
                "device": ["cpu", "musa"],
            },
            self._test_inf_gradients_skip_optim_step,
        )

    def _test_inf_gradients_skip_optim_step(self, device):
        scaler = ShardedGradScaler(init_scale=2.0, enabled=True)
        loss = torch.full((1,), 4.0, dtype=torch.float32, device=device)
        t0 = torch.tensor([float("inf")], dtype=torch.float32, device=device)
        t0.grad = t0.clone()
        opt = torch.optim.SGD([t0], lr=1.0)
        scaler.scale(loss)
        ret_val = scaler.step(opt)
        self.assertTrue(ret_val is None)
