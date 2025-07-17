"""Test mish forward & backward operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing


def mish_forward_inputs():
    return [
        {"inputs": torch.randn((2,), requires_grad=True)},
        {"inputs": torch.randn((32,), requires_grad=True)},
        {"inputs": torch.randn((128,), requires_grad=True)},
        {"inputs": torch.randn((512,), requires_grad=True)},
        {"inputs": torch.randn((1024,), requires_grad=True)},
        {"inputs": torch.randn((16, 1024), requires_grad=True)},
        {"inputs": torch.randn((16, 0), requires_grad=True)},
        {"inputs": torch.randn((0, 0), requires_grad=True)},
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


support_dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    support_dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", mish_forward_inputs())
@pytest.mark.parametrize("dtype", support_dtypes)
def test_mish_fwdbwd(input_data, dtype):
    mish = torch.nn.Mish
    mish_args = {}
    if dtype in [torch.bfloat16, torch.float16]:
        abs_diff, rel_diff = 5e-2, 5e-3
    else:
        abs_diff, rel_diff = 5e-6, 5e-6
    test = testing.OpTest(
        func=mish,
        input_args=mish_args,
        comparators=testing.DefaultComparator(abs_diff, rel_diff),
    )
    test.check_result(
        {
            "input": input_data["inputs"].to(dtype),
        },
        train=True,
    )
