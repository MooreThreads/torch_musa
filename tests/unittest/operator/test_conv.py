"""Test conv2d operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(2, 3, 16, 16, requires_grad=True),
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias": False,
        "in_channels": 3,
        "out_channels": 1,
        "dilation": 1,
        "groups": 1,
    },
    {
        "input": torch.randn(2, 4, 64, 64, requires_grad=True),
        "kernel_size": 5,
        "stride": 1,
        "padding": 1,
        "bias": False,
        "in_channels": 4,
        "out_channels": 1,
        "dilation": 1,
        "groups": 1,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_conv2d(input_data):
    """Test conv2d operators."""
    cpu_input = input_data["input"]
    musa_input = input_data["input"]
    conv2d = torch.nn.Conv2d(
        in_channels=input_data["in_channels"],
        out_channels=input_data["out_channels"],
        kernel_size=input_data["kernel_size"],
        stride=input_data["stride"],
        padding=input_data["padding"],
        dilation=input_data["dilation"],
        groups=input_data["groups"],
        bias=input_data["bias"],
        device="cpu",
    )
    cpu_output = conv2d(cpu_input)
    musa_conv2d = conv2d.to("musa")
    musa_output = musa_conv2d(musa_input.to("musa")) # pylint:disable=E1102
    comparator = testing.DefaultComparator(abs_diff=1e-6)
    assert comparator(cpu_output, musa_output.cpu())
    cpu_output.sum().backward()
    musa_output.sum().backward()
    assert comparator(cpu_input.grad, musa_input.grad.cpu())

    cpu_model = torch.nn.ConvTranspose2d(
        in_channels=input_data["in_channels"],
        out_channels=input_data["out_channels"],
        kernel_size=input_data["kernel_size"],
        stride=input_data["stride"],
        padding=input_data["padding"],
        dilation=input_data["dilation"],
        groups=input_data["groups"],
        bias=input_data["bias"],
        device="cpu",
    )
    cpu_output = cpu_model(cpu_input)
    musa_model = cpu_model.to("musa")
    musa_output = musa_model(musa_input.to("musa")) # pylint:disable=E1102
    comparator = testing.DefaultComparator(abs_diff=1e-6)
    assert comparator(cpu_output, musa_output.cpu())
    cpu_output.sum().backward()
    musa_output.sum().backward()
    assert comparator(cpu_input.grad, musa_input.grad.cpu())
