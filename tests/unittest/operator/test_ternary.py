"""Test binary operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    {
        "input": torch.tensor(random.uniform(-10, 10)),
        "tensor1": torch.randn(30, 30),
        "tensor2": torch.randn(1, 30)
    },
    {
        "input": torch.randn(30),
        "tensor1": torch.tensor(random.uniform(-10, 10)),
        "tensor2": torch.tensor(random.uniform(-10, 10))
    },
    {
        "input": torch.randn(30, 1),
        "tensor1": torch.randn(30, 30),
        "tensor2": torch.randn(1, 30)
    },
    {
        "input": torch.randn(30, 1),
        "tensor1": torch.randn(1, 30),
        "tensor2": torch.randn(30, 1)
    },
]

values = [-1, 0.5, 0, 0.5, 1]
for data in testing.get_raw_data():
    input_datas.append({"input": data, "tensor1": data, "tensor2": data})

all_support_types = testing.get_all_support_types()


def transform_dtype(dtype, value):
    if dtype is torch.float32:
        return float(value)
    if dtype in (torch.int32, torch.int64):
        return int(value)
    return dtype


@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("value", values)
def test_addcmul(input_data, dtype, value):
    input_data["input"] = input_data["input"].to(dtype)
    input_data["tensor1"] = input_data["tensor1"].to(dtype)
    input_data["tensor2"] = input_data["tensor2"].to(dtype)
    input_data["value"] = transform_dtype(dtype, value)
    test = testing.OpTest(func=torch.addcmul, input_args=input_data)
    test.check_result()


# addcdiv only support float32
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("value", values)
def test_addcdiv(input_data, dtype, value):
    input_data["input"] = input_data["input"].to(dtype)
    input_data["tensor1"] = input_data["tensor1"].to(dtype)
    input_data["tensor2"] = abs(input_data["tensor2"].to(dtype)) + 0.001
    input_data["value"] = transform_dtype(dtype, value)
    comparator = testing.DefaultComparator(abs_diff=1e-5)
    test = testing.OpTest(func=torch.addcdiv,
                          input_args=input_data, comparators=comparator)
    test.check_result()


# where not support braodcast in muDNN
input_datas = testing.get_raw_data()
input_datas.append(torch.tensor(random.uniform(-10, 10)))


@pytest.mark.parametrize("data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_where(data, dtype):
    input_args = {}
    input_args["condition"] = data > 0.5
    input_args["input"] = data.to(dtype)
    input_args["other"] = data.to(dtype)
    test = testing.OpTest(func=torch.where, input_args=input_args)
    test.check_result()
