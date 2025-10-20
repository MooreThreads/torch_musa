"""Test glu forward & backward operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
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
