"""Test logit operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest

import torch_musa
from torch_musa import testing

# input, other


def logit_inputs():
    return [
        {"inputs": torch.rand((32,)).requires_grad_(True)},
        {"inputs": torch.rand((16, 1024), requires_grad=True)},
        {"inputs": torch.rand((16, 0), requires_grad=True)},
        {"inputs": torch.rand((16, 16, 1024), requires_grad=True)},
        {
            "inputs": torch.rand((16, 1, 16, 1024), requires_grad=True).to(
                memory_format=torch.channels_last
            )
        },
    ]


float_dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", logit_inputs())
@pytest.mark.parametrize("dtype", float_dtypes)
def test_logit_fwd(input_data, dtype):
    if dtype in [torch.bfloat16, torch.float16]:
        abs_diff, rel_diff = 1e-2, 1e-2
    else:
        abs_diff, rel_diff = 1e-6, 1e-6

    test = testing.OpTest(
        func=torch.logit,
        input_args={},
        comparators=testing.DefaultComparator(abs_diff, rel_diff),
    )
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32(
            {
                "input": input_data["inputs"].to(dtype),
            },
            train=False,
        )
    else:
        test.check_result(
            {
                "input": input_data["inputs"].to(dtype),
            },
            train=False,
        )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", logit_inputs())
@pytest.mark.parametrize("dtype", [torch.float32])
def test_logit_bwd(input_data, dtype):

    abs_diff, rel_diff = 1e-6, 1e-6

    test = testing.OpTest(
        func=torch.logit,
        input_args={},
        comparators=testing.DefaultComparator(abs_diff, rel_diff),
    )
    test.check_result(
        {
            "input": input_data["inputs"].to(dtype),
        },
        train=True,
    )
