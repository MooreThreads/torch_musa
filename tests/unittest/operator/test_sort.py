"""Test sort operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    [torch.tensor(1.5), 0],
    [torch.randn(0), 0],
    [torch.randn(10), 0],
    [torch.randn(10)[2:8], -1],
    [torch.randn(10, 10), 1],
    [torch.randn(10, 0), 0],
    [torch.randn(10, 10).t(), 1],
    [torch.randn(10, 10, 2, 6, 1, 3), 3],
    [torch.randn(10, 4, 1, 1).to(memory_format=torch.channels_last), 0],
]

dtypes = [
    torch.float16,
    torch.float32,
    torch.int32,
    torch.int64,
]

if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("stable", [True])
def test_sort(input_data, dtype, descending, stable):
    input_args = {}
    input_args["input"] = input_data[0].to(dtype)
    dim = input_data[1]
    input_args["dim"] = dim
    input_args["descending"] = descending
    input_args["stable"] = stable
    test = testing.OpTest(func=torch.sort, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
    else:
        test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("stable", [True])
def test_argsort(input_data, dtype, descending, stable):
    input_args = {}
    input_args["input"] = input_data[0].to(dtype)
    dim = input_data[1]
    input_args["dim"] = dim
    input_args["descending"] = descending
    input_args["stable"] = stable
    test = testing.OpTest(func=torch.argsort, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
    else:
        test.check_result()
    test.check_grad_fn()
