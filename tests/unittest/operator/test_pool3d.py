# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name
import torch
import pytest
from torch_musa import testing


input_data = [
    {"input": torch.randn(2, 16, 8, 16, 16, requires_grad=True)},
    {"input": torch.randn(0, 16, 8, 16, 16, requires_grad=True)},
]
kernel_size = [2, 3, (2, 3, 3), (3, 3, 3)]
stride = [1, 3, (2, 1, 1)]
padding = [0, 1, (1, 1, 1)]
dilation = [1, 2, 2]
return_indice = [False, True]
ceil_mode = [False, True]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("kernel_size", kernel_size)
@pytest.mark.parametrize("stride", stride)
@pytest.mark.parametrize("padding", padding)
@pytest.mark.parametrize("dilation", dilation)
@pytest.mark.parametrize("return_indice", return_indice)
@pytest.mark.parametrize("ceil_mode", ceil_mode)
def test_pool3d(
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
    test = testing.OpTest(func=torch.nn.MaxPool3d, input_args=input_params)
    test.check_result({"input": input_data["input"]}, train=True)
    test.check_grad_fn()

    del input_params["dilation"]
    del input_params["return_indices"]
    del input_params["ceil_mode"]
    test = testing.OpTest(func=torch.nn.AvgPool3d, input_args=input_params)
    test.check_result({"input": input_data["input"]}, train=False)
    test.check_grad_fn()


input_data = [
    torch.randn(2, 3, 8, 8, 8).requires_grad_(),
    torch.randn(0, 3, 8, 8, 8).requires_grad_(),
    torch.randn(1, 3, 9, 9, 9).requires_grad_(),
    torch.randn(4, 2, 10, 10, 10).requires_grad_(),
]


pool_params = [
    {"kernel_size": 2, "output_size": (5, 5, 5), "return_indices": False},
    {"kernel_size": (2, 2, 2), "output_size": (6, 6, 6), "return_indices": True},
    {"kernel_size": 2, "output_ratio": 0.5, "return_indices": False},
    {"kernel_size": (2, 2, 2), "output_ratio": (0.7, 0.8, 0.7), "return_indices": True},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("params", pool_params)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fractional_max_pool3d(input_data, params, dtype):
    input_data = input_data.to(dtype)
    batch_size, channel = input_data.shape[:2]
    random_samples = torch.rand(batch_size, channel, 3)
    params["_random_samples"] = random_samples
    m = torch.nn.FractionalMaxPool3d(**params)
    test = testing.OpTest(func=m, input_args={"input": input_data})
    test.check_result()
