# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing

import torch_musa

input_data = [
    {"input": torch.randn(20, 3, 10, 10)},
    {"input": torch.randn(23, 7, 50, 48)[::2, ::3, ::3, ::2]},
    {"input": torch.randn(0, 2, 10, 10)},
    {"input": torch.randn(5, 1, 10, 10).to(memory_format=torch.channels_last)},
    {"input": torch.randn(20, 3, 10, 10).to(memory_format=torch.channels_last)},
    {"input": torch.randn(0, 3, 10, 10).to(memory_format=torch.channels_last)},
]

kernel_size = [2, 3, (3, 2), (3, 3), (5, 5)]

stride = [1, 3, (2, 1)]

padding = [0, 1, (1, 1)]

ceil_mode = [False, True]

count_include_pad = [True, False]

divisor_override = [None, 1, 2]

dilation = [1, 2]
return_indice = [False, True]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("kernel_size", kernel_size)
@pytest.mark.parametrize("stride", stride)
@pytest.mark.parametrize("padding", padding)
@pytest.mark.parametrize("ceil_mode", ceil_mode)
@pytest.mark.parametrize("count_include_pad", count_include_pad)
@pytest.mark.parametrize("divisor_override", divisor_override)
def test_avgpool2d(
    input_data,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    input_params = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
    }
    test = testing.OpTest(
        func=torch.nn.AvgPool2d,
        input_args=input_params,
    )
    test.check_result(input_data)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("kernel_size", kernel_size)
@pytest.mark.parametrize("stride", stride)
@pytest.mark.parametrize("padding", padding)
@pytest.mark.parametrize("dilation", dilation)
@pytest.mark.parametrize("return_indice", return_indice)
@pytest.mark.parametrize("ceil_mode", ceil_mode)
def test_maxpool2d(
    input_data,
    kernel_size,
    stride,
    padding,
    dilation,
    return_indice,
    ceil_mode,
):
    input_params = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indice,
        "ceil_mode": ceil_mode,
    }
    test = testing.OpTest(func=torch.nn.MaxPool2d, input_args=input_params)
    test.check_result(input_data)
    test.check_grad_fn()


input_data = [
    torch.randn(2, 3, 8, 8).requires_grad_(),
    torch.randn(0, 3, 8, 8).requires_grad_(),
    torch.randn(1, 3, 9, 9).requires_grad_(),
    torch.randn(4, 2, 10, 10).requires_grad_(),
]


pool_params = [
    {"kernel_size": 2, "output_size": (5, 5), "return_indices": False},
    {"kernel_size": (2, 2), "output_size": (6, 6), "return_indices": True},
    {"kernel_size": 2, "output_ratio": 0.5, "return_indices": False},
    {"kernel_size": (2, 2), "output_ratio": (0.7, 0.8), "return_indices": True},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("params", pool_params)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fractional_max_pool2d(input_data, params, dtype):
    input_data = input_data.to(dtype)
    batch_size, channel = input_data.shape[:2]
    random_samples = torch.rand(batch_size, channel, 2)
    params["_random_samples"] = random_samples
    m = torch.nn.FractionalMaxPool2d(**params)
    test = testing.OpTest(func=m, input_args={"input": input_data})
    test.check_result(train=True)
