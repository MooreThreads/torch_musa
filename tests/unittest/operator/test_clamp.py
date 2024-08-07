"""Test activation operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random
from typing import Tuple
import copy
import numpy as np
import torch
import pytest
import torch_musa
from torch_musa import testing

# =================================== Test torch.clamp begin =================================== #
input_datas = [
    {
        "input": torch.linspace(-100, 100, 2 * 3 * 4).reshape(2, 3, 4),
        "min": torch.tensor(12),
        "max": torch.tensor(89),
    },
    {
        "input": torch.linspace(-100, 100, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5),
        "min": torch.tensor(12),
        "max": torch.tensor(89),
    },
    {
        "input": torch.linspace(-100, 100, 2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6),
        "min": torch.tensor(12),
        "max": torch.tensor(89),
    },
    {"input": torch.tensor(35), "min": torch.tensor(34), "max": torch.tensor(77)},
    {
        "input": torch.randint(low=-100, high=100, size=[40]),
        "min": torch.linspace(-100, 0, steps=40),
        "max": torch.linspace(0, 100, steps=40),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[4, 5, 6]),
        "min": torch.linspace(-50, 0, 120).reshape(4, 5, 6),
        "max": torch.linspace(0, 50, 120).reshape(4, 5, 6),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[4, 5, 6, 7]),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[4, 5, 6, 7, 8]),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[4, 5, 0, 7, 8]),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[4, 5, 0, 7, 0]),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[4, 5, 6, 7, 8, 2]),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[4, 5, 6, 7, 8, 2, 3]),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[4, 5, 6, 7, 8, 2, 3, 2]),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[2, 1, 3, 4]).to(
            memory_format=torch.channels_last
        ),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[2, 1, 0, 4]).to(
            memory_format=torch.channels_last
        ),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[0, 1, 3, 0]).to(
            memory_format=torch.channels_last
        ),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randint(low=-100, high=100, size=[2, 3, 1, 1]).to(
            memory_format=torch.channels_last
        ),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randn(2, 3, 4, 5, 0, 2, 4),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randn(2, 3, 0, 5).to(memory_format=torch.channels_last),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
    {
        "input": torch.randn(2, 0, 6, 5),
        "min": torch.tensor(-6),
        "max": torch.tensor(95),
    },
]

min_value = [-50, -40]
max_value = [40, 50]

all_dtypes = [torch.float32, torch.int32, torch.int64]
# bf16 is not supported on arch older than qy2
if testing.get_musa_arch() >= 22:
    all_dtypes.append(torch.bfloat16)


def function(input_data, dtype, func, train=False):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    if "out" in input_data.keys() and isinstance(input_data["out"], torch.Tensor):
        input_data["out"] = input_data["out"].to(dtype)
    if "min" in input_data.keys() and isinstance(input_data["min"], torch.Tensor):
        input_data["min"] = input_data["min"].to(dtype)
    if "max" in input_data.keys() and isinstance(input_data["max"], torch.Tensor):
        input_data["max"] = input_data["max"].to(dtype)
    comparator = testing.DefaultComparator(abs_diff=1e-6, equal_nan=True)
    if dtype == torch.bfloat16:
        comparator = testing.DefaultComparator(
            abs_diff=5e-2, rel_diff=5e-3, equal_nan=True
        )
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=comparator,
    )

    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
        test.check_out_ops(fp16=True)
        test.check_grad_fn(fp16=True)
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
        test.check_out_ops(bf16=True)
        test.check_grad_fn(bf16=True)
    else:
        test.check_result(train=train)
        test.check_out_ops()
        test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_min", min_value)
@pytest.mark.parametrize("_max", max_value)
@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("func", [torch.clamp])
def test_clamp_min_max(input_data, _min, _max, dtype, func):
    input_args = {
        "input": input_data["input"],
        "min": _min,
        "max": _max,
    }
    function(input_args, dtype, func)
    input_args.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.clamp.__name__ + "_",
        self_tensor=input_data["input"].to(dtype).clone(),
        input_args=input_args,
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_min", min_value)
@pytest.mark.parametrize("_max", max_value)
def test_clamp_min_max_fp16(input_data, _min, _max):
    input_args = {
        "input": input_data["input"].to(torch.float16).to(torch.float32),
        "min": _min,
        "max": _max,
    }
    test = testing.OpTest(
        func=torch.clamp,
        input_args=input_args,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    test.check_musafp16_vs_musafp32()
    test.check_out_ops(fp16=True)
    test.check_grad_fn(fp16=True)
    input_args.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.clamp.__name__ + "_",
        self_tensor=input_data["input"].to(torch.float16).clone(),
        input_args=input_args,
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_min", min_value)
@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("func", [torch.clamp_min])
def test_clamp_min(input_data, _min, dtype, func):
    input_args = {"input": input_data["input"], "min": _min}
    function(input_args, dtype, func)
    input_args.pop("input")
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=input_data["input"].to(dtype).clone(),
        input_args=input_args,
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_max", max_value)
@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("func", [torch.clamp_max])
def test_clamp_max(input_data, _max, dtype, func):
    input_args = {"input": input_data["input"], "max": _max}
    function(input_args, dtype, func)
    input_args.pop("input")
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=input_data["input"].to(dtype).clone(),
        input_args=input_args,
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_max", max_value)
def test_clamp_max_fp16(input_data, _max):
    input_args = {
        "input": input_data["input"].to(torch.float16).to(torch.float32),
        "max": _max,
    }
    test = testing.OpTest(
        func=torch.clamp_max,
        input_args=input_args,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    test.check_musafp16_vs_musafp32()
    test.check_grad_fn(fp16=True)
    test.check_out_ops(fp16=True)
    input_args.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.clamp_max.__name__ + "_",
        self_tensor=input_data["input"].to(torch.float16).clone(),
        input_args=input_args,
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_min", min_value)
@pytest.mark.parametrize("_max", max_value)
@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("func", [torch.clamp])
def test_clamp_scalar_min_max_out(input_data, _min, _max, dtype, func):
    out = torch.tensor(np.array([]))
    input_args = {"input": input_data["input"], "min": _min, "max": _max, "out": out}
    function(input_args, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("func", [torch.clamp])
def test_clamp_tensor_min_max(input_data, dtype, func):
    function(input_data, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("func", [torch.clamp])
def test_clamp_tensor_min_max_out(input_data, dtype, func):
    out = torch.tensor(np.array([]))
    input_args = {"out": out}
    input_args.update(input_data)
    function(input_args, dtype, func)


# =================================== Test torch.clamp end =================================== #
