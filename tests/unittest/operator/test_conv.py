"""Test conv operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest

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
    {
        "input": torch.randn(2, 1, 64, 64, requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        "kernel_size": 5,
        "stride": 1,
        "padding": 1,
        "bias": False,
        "in_channels": 1,
        "out_channels": 1,
        "dilation": 1,
        "groups": 1,
    },
    {
        "input": torch.randn(0, 1, 64, 64, requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        "kernel_size": 5,
        "stride": 1,
        "padding": 1,
        "bias": False,
        "in_channels": 1,
        "out_channels": 1,
        "dilation": 1,
        "groups": 1,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_conv2d(input_data):
    """Test conv2d operators."""
    conv2d_args = {
        "in_channels": input_data["in_channels"],
        "out_channels": input_data["out_channels"],
        "kernel_size": input_data["kernel_size"],
        "stride": input_data["stride"],
        "padding": input_data["padding"],
        "dilation": input_data["dilation"],
        "groups": input_data["groups"],
        "bias": input_data["bias"],
    }
    test = testing.OpTest(
        func=torch.nn.Conv2d,
        input_args=conv2d_args,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result({"input": input_data["input"]}, train=True)
    test.check_grad_fn()

    test = testing.OpTest(
        func=torch.nn.ConvTranspose2d,
        input_args=conv2d_args,
        comparators=testing.DefaultComparator(abs_diff=2e-6),
    )
    test.check_result({"input": input_data["input"]}, train=True)
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {
            "input": torch.randn(1, 3, 8, 224, 224, requires_grad=True),
            "kernel_size": (3, 3, 3),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "bias": False,
            "in_channels": 3,
            "out_channels": 16,
            "dilation": (1, 1, 1),
            "groups": 1,
        },
        {
            "input": torch.randn(4, 3, 8, 224, 224, requires_grad=True),
            "kernel_size": (3, 7, 7),
            "stride": (2, 2, 2),
            "padding": (3, 3, 3),
            "bias": False,
            "in_channels": 3,
            "out_channels": 64,
            "dilation": (1, 1, 1),
            "groups": 1,
        },
        {
            "input": torch.randn(0, 3, 8, 224, 224, requires_grad=True),
            "kernel_size": (3, 7, 7),
            "stride": (2, 2, 2),
            "padding": (3, 3, 3),
            "bias": False,
            "in_channels": 3,
            "out_channels": 64,
            "dilation": (1, 1, 1),
            "groups": 1,
        },
    ],
)
def test_conv3d(input_data):
    conv3d = torch.nn.Conv3d
    conv3d_args = {
        "in_channels": input_data["in_channels"],
        "out_channels": input_data["out_channels"],
        "kernel_size": input_data["kernel_size"],
        "stride": input_data["stride"],
        "padding": input_data["padding"],
        "dilation": input_data["dilation"],
        "groups": input_data["groups"],
        "bias": input_data["bias"],
    }
    test = testing.OpTest(
        func=conv3d,
        input_args=conv3d_args,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
        #   seed=random.randint(0, 66666)
    )
    test.check_result({"input": input_data["input"]}, train=True)
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {
            "input": torch.randn(20, 16, 50, 100),
            "kernel_size": (3, 5),
            "stride": (2, 1),
            "padding": (4, 2),
            "in_channels": 16,
            "out_channels": 3,
            "dilation": (1, 1),
            "groups": 1,
        }
    ],
)
@pytest.mark.parametrize(
    "bias",
    [True, False],
)
@pytest.mark.parametrize(
    "requires_grad",
    [True, False],
)
def test_conv2d_transpose(input_data, bias, requires_grad):
    func_args = {
        "in_channels": input_data["in_channels"],
        "out_channels": input_data["out_channels"],
        "kernel_size": input_data["kernel_size"],
        "stride": input_data["stride"],
        "padding": input_data["padding"],
        "dilation": input_data["dilation"],
        "groups": input_data["groups"],
        "bias": bias,
    }
    test = testing.OpTest(
        func=torch.nn.ConvTranspose2d,
        input_args=func_args,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    with torch.set_grad_enabled(requires_grad):
        func_input = input_data["input"]
        func_input.requires_grad_(requires_grad)
        test.check_result({"input": func_input}, train=requires_grad)
    test.check_grad_fn()
