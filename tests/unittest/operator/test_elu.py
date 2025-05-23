"""Test elu forward & backward operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest

import torch_musa
from torch_musa import testing

# input, other


def elu_forward_inputs():
    return [
        {"inputs": torch.randn((2,), requires_grad=True)},
        {"inputs": torch.randn((32,), requires_grad=True)},
        {"inputs": torch.randn((128,), requires_grad=True)},
        {"inputs": torch.randn((512,), requires_grad=True)},
        {"inputs": torch.randn((1024,), requires_grad=True)},
        {"inputs": torch.randn((16, 1024), requires_grad=True)},
        {"inputs": torch.randn((16, 0), requires_grad=True)},
        {"inputs": torch.randn((0, 0), requires_grad=True)},
        {"inputs": torch.randn((16, 16, 1024), requires_grad=True)},
        {"inputs": torch.randn((16, 16, 16, 1024), requires_grad=True)},
        {
            "inputs": torch.randn((16, 1, 16, 1024), requires_grad=True).to(
                memory_format=torch.channels_last
            )
        },
        {
            "inputs": torch.randn((16, 12, 1, 1), requires_grad=True).to(
                memory_format=torch.channels_last
            )
        },
        {
            "inputs": torch.randn((16, 0, 1, 1), requires_grad=True).to(
                memory_format=torch.channels_last
            )
        },
        {
            "inputs": torch.randn((0, 0, 1, 1), requires_grad=True).to(
                memory_format=torch.channels_last
            )
        },
        {
            "inputs": torch.randn((16, 12, 5, 2), requires_grad=True).to(
                memory_format=torch.channels_last
            )
        },
    ]


float_dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", elu_forward_inputs())
@pytest.mark.parametrize("dtype", float_dtypes)
def test_elu_fwd(input_data, dtype):
    elu = torch.nn.ELU
    elu_args = {}
    if dtype in [torch.bfloat16, torch.float16]:
        abs_diff, rel_diff = 1e-3, 1e-3
    else:
        abs_diff, rel_diff = 1e-6, 1e-6

    test = testing.OpTest(
        func=elu,
        input_args=elu_args,
        comparators=testing.DefaultComparator(abs_diff, rel_diff),
    )
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32(
            {
                "input": input_data["inputs"].to(dtype),
            },
            train=True,
        )
    else:
        test.check_result(
            {
                "input": input_data["inputs"].to(dtype),
            },
            train=True,
        )
    # We only care about grad_fn, so we use fp32 for computation on the CPU.
    assert (
        elu()(input_data["inputs"].to(dtype).musa().requires_grad_()).grad_fn.__class__
        == elu()(
            input_data["inputs"].to(torch.float32).cpu().requires_grad_()
        ).grad_fn.__class__
    )
