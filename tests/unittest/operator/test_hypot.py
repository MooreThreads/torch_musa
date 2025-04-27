"""Test hypot forward & backward operator."""

import torch
import pytest
from torch_musa import testing


# input, other
def hypot_forward_inputs():
    return [
        {
            "inputs": (
                torch.randn((2,), requires_grad=True),
                torch.randn((2,), requires_grad=True),
            )
        },
        {
            "inputs": (
                torch.randn((32,), requires_grad=True),
                torch.randn((32,), requires_grad=True),
            )
        },
        {
            "inputs": (
                torch.randn((128,), requires_grad=True),
                torch.randn((128,), requires_grad=True),
            )
        },
        {
            "inputs": (
                torch.randn((512,), requires_grad=True),
                torch.randn((512,), requires_grad=True),
            )
        },
        {
            "inputs": (
                torch.randn((1024,), requires_grad=True),
                torch.randn((1024,), requires_grad=True),
            )
        },
        {
            "inputs": (
                torch.randn((16, 1024), requires_grad=True),
                torch.randn((16, 1024), requires_grad=True),
            )
        },
        {
            "inputs": (
                torch.randn((16, 16, 1024), requires_grad=True),
                torch.randn((16, 16, 1024), requires_grad=True),
            )
        },
        {
            "inputs": (
                torch.randn((16, 16, 16, 1024), requires_grad=True),
                torch.randn((16, 16, 16, 1024), requires_grad=True),
            )
        },
        {
            "inputs": (
                torch.randn((16, 16, 0, 1024), requires_grad=True),
                torch.randn((16, 16, 0, 1024), requires_grad=True),
            )
        },
        {
            "inputs": (
                torch.randn((16, 4, 16, 1024), requires_grad=True).to(
                    memory_format=torch.channels_last
                ),
                torch.randn((16, 4, 16, 1024), requires_grad=True).to(
                    memory_format=torch.channels_last
                ),
            )
        },
        {
            "inputs": (
                torch.randn((16, 18, 1, 1024), requires_grad=True).to(
                    memory_format=torch.channels_last
                ),
                torch.randn((16, 18, 1, 1024), requires_grad=True).to(
                    memory_format=torch.channels_last
                ),
            )
        },
        {
            "inputs": (
                torch.randn((16, 16, 16, 1024), requires_grad=True).to(
                    memory_format=torch.channels_last
                ),
                torch.randn((16, 16, 16, 1024), requires_grad=True).to(
                    memory_format=torch.channels_last
                ),
            )
        },
    ]


support_dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    support_dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", hypot_forward_inputs())
@pytest.mark.parametrize("dtype", support_dtypes)
def test_hypot(input_data, dtype):
    """test hypot operation"""
    hypot = torch.hypot
    hypot_args = {}
    input1, input2 = input_data["inputs"]
    input1 = input1.to(dtype)
    input2 = input2.to(dtype)
    test = testing.OpTest(
        func=hypot,
        input_args=hypot_args,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )

    test.check_result(
        {
            "input": input1,
            "other": input2,
        },
        train=True,
    )
    cpu_output = torch.hypot(
        input1.to(dtype).cpu().requires_grad_(), input2.to(dtype).cpu().requires_grad_()
    )
    musa_output = torch.hypot(
        input1.to(dtype).to("musa").requires_grad_(),
        input2.to(dtype).to("musa").requires_grad_(),
    )
    assert cpu_output.grad_fn.__class__ == musa_output.grad_fn.__class__
