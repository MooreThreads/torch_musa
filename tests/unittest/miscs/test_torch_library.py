"""test torch library"""

# pylint: disable=unused-import,abstract-method,arguments-differ
import pytest

import torch
import torch_musa


class Gelu(torch.autograd.Function):
    """Gelu forward and backward implementation"""

    @staticmethod
    def forward(ctx, x, approximate):
        if approximate == "tanh":
            out = (
                0.5
                * x
                * (1 + torch.tanh(x * 0.79788456 * (1 + 0.044715 * torch.pow(x, 2))))
            )
        else:
            raise NotImplementedError
        ctx.save_for_backward(x)
        ctx.approximate = approximate

        return out

    @staticmethod
    def backward(ctx, out_grad):
        (inp,) = ctx.saved_tensors
        approximate = ctx.approximate
        if approximate == "tanh":
            tanh_out = torch.tanh(0.79788456 * inp * (1 + 0.044715 * torch.pow(inp, 2)))
            partial_d = 0.5 * inp * (
                (1 - torch.pow(tanh_out, 2))
                * (0.79788456 + 0.1070322243 * torch.pow(inp, 2))
            ) + 0.5 * (1 + tanh_out)
            return partial_d * out_grad, None

        raise NotImplementedError


@pytest.mark.parametrize("dispatch_key", ["MUSA", "PrivateUse1"])
def test_musa_dispatch_key(dispatch_key):
    """test new library to register new operators"""
    ns_ = f"{dispatch_key}_lib"
    musa_lib = torch.library.Library(ns_, "DEF")
    musa_lib.define("relu_add_one(Tensor self) -> Tensor")

    def relu_add_one_impl(self):
        return self.relu() + 1.0

    musa_lib.impl("relu_add_one", relu_add_one_impl, dispatch_key)

    x = torch.randn((1024), device="musa")
    out = getattr(torch.ops, ns_).relu_add_one(x)
    assert torch.allclose(out, x.relu() + 1.0)


@pytest.mark.parametrize("dispatch_key", ["AutogradMUSA", "AutogradPrivateUse1"])
def test_musa_autograd_dispatch_key(dispatch_key):
    """test op registration on Autograd dispatch key"""

    def gelu(x, *, approximate="none"):
        return Gelu.apply(x, approximate)

    ns_ = f"{dispatch_key}_lib"
    musa_lib = torch.library.Library(ns_, "DEF")
    musa_lib.define("gelu_musa(Tensor self, *, str approximate='none') -> Tensor")
    musa_lib.impl("gelu_musa", gelu, dispatch_key)

    torch.musa.manual_seed(42)
    x = torch.randn((1024,), device="musa", requires_grad=True)
    torch.musa.manual_seed(42)
    x_golden = torch.randn((1024,), device="musa", requires_grad=True)

    out = getattr(torch.ops, ns_).gelu_musa(x, approximate="tanh")
    out.sum().backward()
    out_golden = torch.nn.functional.gelu(x_golden, approximate="tanh")
    out_golden.sum().backward()

    assert torch.allclose(out, out_golden, rtol=1e-5, atol=1e-5)
    assert torch.allclose(x.grad, x_golden.grad, rtol=1e-5, atol=1e-5)
