"""Test addmv operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import copy
import torch
import pytest
import torch_musa
from torch_musa.testing import get_musa_arch
from torch_musa import testing

input_data = [
    {
        "input": torch.randn(4),
        "mat": torch.randn(4, 0),
        "vec": torch.randn(0),
        "beta": 1,
        "alpha": 1.2,
    },
    {
        "input": torch.randn(0),
        "mat": torch.randn(0, 30),
        "vec": torch.randn(30),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(2),
        "mat": torch.randn(2, 30),
        "vec": torch.randn(30),
        "beta": 1,
        "alpha": 1.2,
    },
    {
        "input": torch.tensor(2.0),
        "mat": torch.randn(16, 8),
        "vec": torch.randn(8),
        "beta": 1.5,
        "alpha": 1.2,
    },
    {
        "input": torch.randn(1),
        "mat": torch.randn(30, 5).t(),
        "vec": torch.randn(30),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(5),
        "mat": torch.randn(30, 5).t(),
        "vec": torch.randn(30),
        "beta": 1.2,
        "alpha": 1.7,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_addmv(input_data):
    test = testing.OpTest(
        func=torch.addmv,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_addmv_fp16(input_data):
    test = testing.OpTest(
        func=torch.addmv,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3),
    )
    test.check_musafp16_vs_musafp32()
    test.check_out_ops(fp16=True)
    test.check_grad_fn(fp16=True)


@pytest.mark.skipif(get_musa_arch() <= 21, reason="Only support arch greater equal 21")
@pytest.mark.parametrize("input_data", input_data)
def test_addmv_bf16(input_data):
    test = testing.OpTest(
        func=torch.addmv,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1, rel_diff=2e-2),
    )
    test.check_musabf16_vs_musafp16()
    test.check_out_ops(bf16=True)
    test.check_grad_fn(bf16=True)


inplace_input_data = [
    {
        "input": torch.randn(4),
        "mat": torch.randn(4, 0),
        "vec": torch.randn(0),
        "beta": 1,
        "alpha": 1.2,
    },
    {
        "input": torch.randn(0),
        "mat": torch.randn(0, 30),
        "vec": torch.randn(30),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(10),
        "mat": torch.randn(10, 30),
        "vec": torch.randn(30),
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
def test_addmv_inplace(input_data, dtype):
    input_data_tmp = copy.deepcopy(input_data)
    self_ = input_data_tmp["input"].to(dtype)
    input_data_tmp["mat"] = input_data_tmp["mat"].to(dtype)
    input_data_tmp["vec"] = input_data_tmp["vec"].to(dtype)

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
        func_name=torch.addmv.__name__ + "_",
        self_tensor=self_,
        input_args=input_data_tmp,
        comparators=[comparator],
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)


out_input_data = [
    {
        "input": torch.randn(128),
        "mat": torch.randn(128, 256),
        "vec": torch.randn(256),
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
def test_addmv_out(out_input_data, dtype):
    # get musa res
    musa_input_data = {}
    for key in out_input_data:
        if isinstance(out_input_data[key], torch.Tensor):
            musa_input_data[key] = out_input_data[key].clone().to("musa")
        else:
            musa_input_data[key] = out_input_data[key]
    musa_input_data["out"] = musa_input_data["input"].clone()
    musa_res = torch.addmv(**musa_input_data)
    # get musa res
    out_input_data["out"] = out_input_data["input"]
    cpu_res = torch.addmv(**out_input_data)

    if dtype == torch.bfloat16:
        comparator = testing.DefaultComparator(
            abs_diff=1, rel_diff=2e-2, equal_nan=True
        )
    else:
        comparator = testing.DefaultComparator(
            abs_diff=5e-2, rel_diff=5e-3, equal_nan=True
        )
    assert comparator(cpu_res, musa_res)
