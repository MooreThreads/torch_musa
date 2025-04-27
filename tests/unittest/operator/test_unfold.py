# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing

import torch_musa


dtypes = testing.get_float_types()
input_datas = [
    torch.randn(64, 29, 27),
    torch.randn(16, 32, 32, 32),
    torch.randn(1, 8, 16, 16),
    torch.randn(64, 2, 17, 17),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("kernel_size", [2, 3, 5])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_unfold(input_data, kernel_size, dilation, padding, stride, dtype):
    input_args = {
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
    }
    data = input_data.to(dtype)
    test = testing.OpTest(func=torch.nn.Unfold, input_args=input_args)
    test.check_result({"input": data})
