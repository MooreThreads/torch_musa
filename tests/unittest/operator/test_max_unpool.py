"""Test for torch.nn.MaxUnpool2d operator in MUSA backend."""

# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name
import torch
import pytest
from torch_musa import testing


input_data = [
    {"input": torch.randn(20, 3, 10, 10)},
    {"input": torch.randn(23, 7, 50, 48)[::2, ::3, ::3, ::2]},
    {"input": torch.randn(0, 2, 10, 10)},
    {"input": torch.randn(5, 1, 10, 10).to(memory_format=torch.channels_last)},
    {"input": torch.randn(20, 3, 10, 10).to(memory_format=torch.channels_last)},
    {"input": torch.randn(0, 3, 10, 10).to(memory_format=torch.channels_last)},
]

kernel_size = [2, 3, (3, 2), (3, 3), (5, 5)]
stride = [2, 3, (2, 1)]
padding = [0, 1, (1, 1)]
ceil_mode = [False, True]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("kernel_size", kernel_size)
@pytest.mark.parametrize("stride", stride)
@pytest.mark.parametrize("padding", padding)
@pytest.mark.parametrize("ceil_mode", ceil_mode)
def test_max_unpool_2d(input_data, kernel_size, stride, padding, ceil_mode):
    max_pool2d_params = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "dilation": 1,
        "return_indices": True,
    }

    pool = torch.nn.MaxPool2d(**max_pool2d_params)
    output, indices = pool(input_data["input"])

    max_unpool2d_params = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
    }

    test = testing.OpTest(
        func=torch.nn.MaxUnpool2d,
        input_args=max_unpool2d_params,
    )

    max_unpool2d_input_data = {
        "input": output,
        "indices": indices,
        "output_size": input_data["input"].size(),
    }

    test.check_result(max_unpool2d_input_data)
