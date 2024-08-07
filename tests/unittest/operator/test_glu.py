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
