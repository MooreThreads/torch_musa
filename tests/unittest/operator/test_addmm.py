"""Test addmm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import copy
import torch
import pytest
import torch_musa
from torch_musa.testing import get_musa_arch
from torch_musa import testing

input_data = [
    {
        "input": torch.randn(4, 2),
        "mat1": torch.randn(4, 0),
        "mat2": torch.randn(0, 2),
        "beta": 1,
        "alpha": 1.2,
    },
    {
        "input": torch.randn(0, 2),
        "mat1": torch.randn(0, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(2, 2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(2, 2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.7,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(2, 2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.tensor(112.5),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(5, 1),
        "mat1": torch.randn(30, 5).t(),
        "mat2": torch.randn(30, 5),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(1, 5),
        "mat1": torch.randn(5, 30),
        "mat2": torch.randn(30, 5),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(1, 5),
        "mat1": torch.randn(30, 5).t(),
        "mat2": torch.randn(5, 30).t(),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(1, 5),
        "mat1": torch.randn(5, 30),
        "mat2": torch.randn(5, 30).t(),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(128, 128),
        "mat1": torch.randn(128, 128),
        "mat2": torch.randn(128, 128),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(128, 64),
        "mat1": torch.randn(128, 256),
        "mat2": torch.randn(256, 64),
        "beta": 1.2,
        "alpha": 1.7,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_addmm(input_data):
    test = testing.OpTest(
        func=torch.addmm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_addmm_fp16(input_data):
    test = testing.OpTest(
        func=torch.addmm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3),
    )
    test.check_musafp16_vs_musafp32()
    test.check_out_ops(fp16=True)
    test.check_grad_fn(fp16=True)


@pytest.mark.skipif(get_musa_arch() <= 21, reason="Only support arch greater equal 21")
@pytest.mark.parametrize("input_data", input_data)
def test_addmm_bf16(input_data):
    test = testing.OpTest(
        func=torch.addmm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1, rel_diff=2e-2),
    )
    test.check_musabf16_vs_musafp16()
    test.check_out_ops(bf16=True)
    test.check_grad_fn(bf16=True)


inplace_input_data = [
    {
        "input": torch.randn(4, 2),
        "mat1": torch.randn(4, 0),
        "mat2": torch.randn(0, 2),
        "beta": 1,
        "alpha": 1.2,
    },
    {
        "input": torch.randn(0, 2),
        "mat1": torch.randn(0, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(2, 2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(2, 2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.7,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(2, 2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(128, 128),
        "mat1": torch.randn(128, 128),
        "mat2": torch.randn(128, 128),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(128, 64),
        "mat1": torch.randn(128, 256),
        "mat2": torch.randn(256, 64),
        "beta": 1.2,
        "alpha": 1.7,
    },
]

inplace_data_type = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    inplace_data_type.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inplace_input_data)
@pytest.mark.parametrize("dtype", inplace_data_type)
def test_addmm_inplace(input_data, dtype):
    input_data_tmp = copy.deepcopy(input_data)
    self_ = input_data_tmp["input"].to(dtype)
    input_data_tmp["mat1"] = input_data_tmp["mat1"].to(dtype)
    input_data_tmp["mat2"] = input_data_tmp["mat2"].to(dtype)

    input_data_tmp.pop("input")
    if dtype == torch.bfloat16:
        comparator = testing.DefaultComparator(
            abs_diff=1, rel_diff=2e-2, equal_nan=True
        )
    else:
        comparator = testing.DefaultComparator(
            abs_diff=5e-2, rel_diff=5e-3, equal_nan=True
        )
    test = testing.InplaceOpChek(
        func_name=torch.addmm.__name__ + "_",
        self_tensor=self_,
        input_args=input_data_tmp,
        comparators=[comparator],
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)


out_input_data = [
    {
        "input": torch.randn(128, 64),
        "mat1": torch.randn(128, 256),
        "mat2": torch.randn(256, 64),
        "beta": 1.2,
        "alpha": 1.7,
    }
]
out_data_type = [torch.float32]
if testing.get_musa_arch() >= 22:
    out_data_type.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("out_input_data", out_input_data)
@pytest.mark.parametrize("dtype", out_data_type)
def test_addmm_out(out_input_data, dtype):
    # get musa res
    musa_input_data = {}
    for key in out_input_data:
        if isinstance(out_input_data[key], torch.Tensor):
            musa_input_data[key] = out_input_data[key].clone().to("musa")
        else:
            musa_input_data[key] = out_input_data[key]
    musa_input_data["out"] = musa_input_data["input"]
    musa_res = torch.addmm(**musa_input_data)
    # get musa res
    out_input_data["out"] = out_input_data["input"]
    cpu_res = torch.addmm(**out_input_data)

    if dtype == torch.bfloat16:
        comparator = testing.DefaultComparator(
            abs_diff=1, rel_diff=2e-2, equal_nan=True
        )
    else:
        comparator = testing.DefaultComparator(
            abs_diff=5e-2, rel_diff=5e-3, equal_nan=True
        )
    assert comparator(cpu_res, musa_res)
