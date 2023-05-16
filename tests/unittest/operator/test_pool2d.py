# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing

import torch_musa

input_data = [{'input': torch.randn(20, 3, 10, 10)},]

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
    input_data, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    input_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'ceil_mode': ceil_mode,
            'count_include_pad': count_include_pad,
            'divisor_override': divisor_override}
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
