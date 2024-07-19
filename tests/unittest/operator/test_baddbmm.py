"""Test baddbmm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import copy
import torch
import pytest
import torch_musa
from torch_musa import testing

input_data = [
    {
        "input": torch.randn(2),
        "batch1": torch.randn(4, 5, 0),
        "batch2": torch.randn(4, 0, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(4, 5, 2),
        "batch1": torch.randn(4, 5, 0),
        "batch2": torch.randn(4, 0, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(5, 2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": 2.4,
        "alpha": 3.2,
    },
    {
        "input": torch.randn(4, 5, 2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": 2.0,
        "alpha": 1.0,
    },
    {
        "input": torch.randn(2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": -2.4,
        "alpha": 3.2,
    },
]
dtypes = [torch.float16, torch.float32]


@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", dtypes)
def test_baddbmm(input_data, dtype):
    input_data["input"] = input_data["input"].to(dtype)
    input_data["batch1"] = input_data["batch1"].to(dtype)
    input_data["batch2"] = input_data["batch2"].to(dtype)
    if dtype == torch.float16:
        test = testing.OpTest(
            func=torch.baddbmm,
            input_args=input_data,
            comparators=testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3),
        )
        test.check_musafp16_vs_musafp32()
        test.check_out_ops(fp16=True)
        test.check_grad_fn(fp16=True)
    else:
        test = testing.OpTest(
            func=torch.baddbmm,
            input_args=input_data,
            comparators=testing.DefaultComparator(abs_diff=1e-3),
        )
        test.check_result()
        test.check_out_ops()
        test.check_grad_fn()


inplace_input_data = [
    {
        "input": torch.randn(4, 5, 2),
        "batch1": torch.randn(4, 5, 0),
        "batch2": torch.randn(4, 0, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(4, 5, 2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(4, 5, 2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": 2.4,
        "alpha": 3.2,
    },
    {
        "input": torch.randn(4, 5, 2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": 2.0,
        "alpha": 1.0,
    },
    {
        "input": torch.randn(4, 5, 2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": -2.4,
        "alpha": 3.2,
    },
]
dtypes = [torch.float16, torch.float32]


@pytest.mark.parametrize("input_data_", inplace_input_data)
@pytest.mark.parametrize("dtype", dtypes)
def test_baddbmm_inplace(input_data_, dtype):
    input_data = copy.deepcopy(input_data_)
    input_data["input"] = input_data["input"].to(dtype)
    input_data["batch1"] = input_data["batch1"].to(dtype)
    input_data["batch2"] = input_data["batch2"].to(dtype)
    self_tensor = input_data["input"]
    input_data.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.baddbmm.__name__ + "_",
        self_tensor=self_tensor,
        input_args=input_data,
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)
