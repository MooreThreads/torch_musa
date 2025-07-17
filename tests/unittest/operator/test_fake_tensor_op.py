"""Test operators that registered FakeTensor implementation"""

# pylint: disable=C0115, C0116
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor


class TestFakeTensorImpl:
    def assert_is_fake_tensor(self, t):
        assert isinstance(t, FakeTensor)

    def test_fused_rmsnorm(self):
        device = torch.musa.current_device()
        x = torch.randn((2, 128, 4096), device=device)
        weight = torch.randn((4096,), device=device)

        with FakeTensorMode(allow_non_fake_inputs=True):
            out = torch.rms_norm(x, (4096,), weight, 1e-6)
        self.assert_is_fake_tensor(out)

        with FakeTensorMode():
            x = torch.randn((2, 128, 4096), device=device)
            weight = torch.randn((4096,), device=device)
            out = torch.rms_norm(x, (4096,), weight, 1e-6)
        self.assert_is_fake_tensor(out)
        assert out.device.type == "musa"
        assert out.shape == (2, 128, 4096)

    def test_fused_rmsnorm_bwd(self):
        device = torch.musa.current_device()

        with FakeTensorMode():
            inp = torch.randn((2, 128, 4096), device=device)
            weight = torch.randn((4096,), device=device)
            grad_out = torch.randn((2, 128, 4096), device=device)
            invvar = torch.randn((2, 128), device=device)

            grad_input, grad_weight = torch.ops.aten._fused_rmsnorm_backward(
                grad_out, invvar, inp, (4096,), 1e-6, weight
            )

        self.assert_is_fake_tensor(grad_input)
        self.assert_is_fake_tensor(grad_weight)
        assert grad_input.shape == (2, 128, 4096)
        assert grad_weight.shape == (4096,)
