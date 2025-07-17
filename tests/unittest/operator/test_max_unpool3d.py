"""Test for torch.nn.MaxUnpool3d operator in MUSA backend."""

# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name
import torch
import pytest
from torch_musa import testing


input_data = [
    {"input": torch.randn(2, 16, 8, 16, 16)},
    {"input": torch.randn(0, 16, 8, 16, 16)},
    {"input": torch.randn(0, 16, 5, 9, 9)},
]

kernel_size = [2, 3, (2, 3, 3), (3, 3, 3)]
stride = [1, 2, (2, 2, 1)]
padding = [0, 1, (1, 1, 1)]
ceil_mode = [False, True]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("kernel_size", kernel_size)
@pytest.mark.parametrize("stride", stride)
@pytest.mark.parametrize("padding", padding)
@pytest.mark.parametrize("ceil_mode", ceil_mode)
def test_max_unpool_3d(input_data, kernel_size, stride, padding, ceil_mode):
    max_pool3d_params = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "dilation": 1,
        "return_indices": True,
    }

    pool3d = torch.nn.MaxPool3d(**max_pool3d_params)
    output, indices = pool3d(input_data["input"])

    max_unpool3d_params = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
    }

    test = testing.OpTest(
        func=torch.nn.MaxUnpool3d,
        input_args=max_unpool3d_params,
    )

    max_unpool3d_input_data = {
        "input": output,
        "indices": indices,
        "output_size": input_data["input"].size(),
    }

    test.check_result(max_unpool3d_input_data)
