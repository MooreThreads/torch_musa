"""Test glu forward & backward operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, C0103
import torch
from torch.nn import functional as F
import pytest

import torch_musa
from torch_musa import testing

# input, other


def glu_forward_inputs():
    return [
        {"inputs": torch.randn((2,), requires_grad=True)},
        {"inputs": torch.randn((32,), requires_grad=True)},
        {"inputs": torch.randn((128,), requires_grad=True)},
        {"inputs": torch.randn((512,), requires_grad=True)},
        {"inputs": torch.randn((1024,), requires_grad=True)},
        {"inputs": torch.randn((16, 1024), requires_grad=True)},
        {"inputs": torch.randn((16, 16, 1024), requires_grad=True)},
        {"inputs": torch.randn((16, 16, 16, 1024), requires_grad=True)},
        {"inputs": torch.randn((16, 16, 0, 1024), requires_grad=True)},
        {
            "inputs": torch.randn((16, 4, 16, 1024), requires_grad=True).to(
                memory_format=torch.channels_last
            )
        },
        {
            "inputs": torch.randn((16, 18, 1, 1024), requires_grad=True).to(
                memory_format=torch.channels_last
            )
        },
        {
            "inputs": torch.randn((16, 16, 16, 1024), requires_grad=True).to(
                memory_format=torch.channels_last
            )
        },
    ]


def glu_backward_inputs():
    return [
        torch.randn((16, 32), requires_grad=False),
        torch.randn((8, 64), requires_grad=False),
        torch.randn((4, 128), requires_grad=False),
        torch.randn((2, 256), requires_grad=False),
    ]


support_dtypes = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", glu_forward_inputs())
@pytest.mark.parametrize("dtype", support_dtypes)
def test_glu_fwdbwd(input_data, dtype):
    glu = torch.nn.GLU
    glu_args = {}
    test = testing.OpTest(
        func=glu,
        input_args=glu_args,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result(
        {
            "input": input_data["inputs"].to(dtype),
        },
        train=True,
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("x", glu_backward_inputs())
@pytest.mark.parametrize("dtype", support_dtypes)
def test_glu_backward_out(x, dtype):
    x = x.to(dtype).to("musa")
    x.requires_grad_(False)

    y = torch.nn.functional.glu(x, dim=1)
    grad_output = torch.ones_like(y)

    ref_grad_input = torch.ops.aten.glu_backward(grad_output, x, 1)

    # out variant
    out = torch.empty_like(x)
    with torch.no_grad():
        torch.ops.aten.glu_backward.grad_input(grad_output, x, 1, grad_input=out)

    torch.testing.assert_close(out, ref_grad_input, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [[32, 64], [8, 64, 128], [2, 2, 16, 32]])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_glu_jvp(shape, dtype):
    cpu_x = torch.randn(*shape, dtype=dtype)
    cpu_dx = torch.randn_like(cpu_x)

    musa_x = cpu_x.musa()
    musa_dx = cpu_dx.musa()

    def f(x, dx):
        y = torch.nn.functional.glu(x, -1)
        z = torch.ops.aten.glu_jvp(y, x, dx, -1)
        return z.cpu()

    cpu_out = f(cpu_x, cpu_dx)
    musa_out = f(musa_x, musa_dx)

    torch.testing.assert_close(cpu_out, musa_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [[32, 64], [8, 64, 128], [2, 2, 16, 32]])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_glu_backward_jvp(shape, dtype):
    cpu_x = torch.randn(*shape, dtype=dtype)
    cpu_dx = torch.randn_like(cpu_x)
    cpu_gx = torch.randn_like(cpu_x)

    musa_x = cpu_x.musa()
    musa_dx = cpu_dx.musa()
    musa_gx = cpu_gx.musa()

    cpu_y = torch.nn.functional.glu(cpu_x, -1)
    cpu_dy = torch.randn_like(cpu_y)
    cpu_gy = torch.randn_like(cpu_y)
    del cpu_y

    musa_dy = cpu_dy.musa()
    musa_gy = cpu_gy.musa()

    f = torch.ops.aten.glu_backward_jvp
    cpu_out = f(cpu_gx, cpu_gy, cpu_x, cpu_dy, cpu_dx, -1)
    musa_out = f(musa_gx, musa_gy, musa_x, musa_dy, musa_dx, -1).cpu()

    torch.testing.assert_close(cpu_out, musa_out, rtol=1e-5, atol=1e-5)
