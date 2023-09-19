"""Test binary operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, C3001, W0622
import random
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    {
        "input": torch.tensor(random.uniform(-10, 10)),
        "tensor1": torch.randn(30, 30),
        "tensor2": torch.randn(1, 30),
    },
    {
        "input": torch.randn(30),
        "tensor1": torch.tensor(random.uniform(-10, 10)),
        "tensor2": torch.tensor(random.uniform(-10, 10)),
    },
    {
        "input": torch.randn(30, 1),
        "tensor1": torch.randn(30, 30),
        "tensor2": torch.randn(1, 30),
    },
    {
        "input": torch.randn(30, 1),
        "tensor1": torch.randn(1, 30),
        "tensor2": torch.randn(30, 1),
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


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("value", values)
def test_addcmul(input_data, dtype, value):
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": input_data["tensor2"].to(dtype),
        "value": transform_dtype(dtype, value),
    }
    test = testing.OpTest(func=torch.addcmul, input_args=input_dict)
    test.check_result()


# addcdiv only support float32
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("value", values)
def test_addcdiv(input_data, dtype, value):
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": torch.abs(input_data["tensor2"].to(dtype)) + 0.001,
        "value": transform_dtype(dtype, value),
    }
    comparator = testing.DefaultComparator(abs_diff=1e-5)
    test = testing.OpTest(
        func=torch.addcdiv, input_args=input_dict, comparators=comparator
    )
    test.check_result()


# where not support braodcast in muDNN
input_datas = testing.get_raw_data()
input_datas.append(torch.tensor(random.uniform(-10, 10)))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_where(data, dtype):
    input_args = {}
    input_args["condition"] = data > 0.5
    input_args["input"] = data.to(dtype)
    input_args["other"] = data.to(dtype)
    test = testing.OpTest(func=torch.where, input_args=input_args)
    test.check_result()


dtypes = [torch.float32, torch.half, torch.uint8, torch.int8, torch.int16, torch.int32]
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("shape", [(3, 2), (200, 100)])
@pytest.mark.parametrize("dtype", dtypes)
def test_naive_where(dtype, shape):
    "Testing naive where : error with int64"
    input = torch.randint(-5, 5, size=shape).to(dtype)
    output_cpu = torch.where(input)
    output_musa = torch.where(input.to("musa"))
    comparator = lambda musa, cpu: (musa == cpu).all()
    for musa, cpu in zip(output_musa, output_cpu):
        assert comparator(musa, cpu)
