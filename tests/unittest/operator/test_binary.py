"""Test binary operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    {"input": torch.tensor(random.uniform(-10, 10)), "other": torch.randn(30, 30)},
    {"input": torch.randn(30), "other": torch.tensor(random.uniform(-10,10))},
    {"input": torch.randn(30, 1), "other": torch.randn(30, 30)},
    {"input": torch.randn(30, 1), "other": torch.randn(1, 30)},
]

for data in testing.get_raw_data():
    input_datas.append({"input": data, "other": data})

all_funcs_except_div = [
    torch.add,
    torch.sub,
    torch.mul,
    torch.eq,
    torch.ne,
    torch.gt,
    torch.ge,
    torch.greater_equal,
    torch.greater,
]

all_support_types = testing.get_all_support_types()

def function(input_data, dtype, other_dtype,func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    if isinstance(input_data["other"], torch.Tensor):
        input_data["other"] = input_data["other"].to(other_dtype)
    if func in (torch.div, torch.remainder):
        input_data["other"] = abs(input_data["other"]) + 0.0001
    test = testing.OpTest(func=func, input_args=input_data)
    test.check_result()


# normal case
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("other_dtype", all_support_types)
@pytest.mark.parametrize("func", all_funcs_except_div)
def test_binary(input_data, dtype, other_dtype, func):
    function(input_data, dtype, other_dtype, func)


# test div, remainder which only support float and make sure other is not zero
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("func", [torch.div, torch.remainder])
def test_div(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


# test add_alpha and sub_alpha(only support float)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn(30), "other": torch.randn(30), "alpha": random.uniform(-10, 10)},
        {"input": torch.randn(30), "other": torch.tensor(random.uniform(-100, 100)),
            "alpha": random.uniform(-10, 10)},
        {"input": random.uniform(0, 1), "other": random.uniform(-1, 1),
            "alpha": random.uniform(-1, 1)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("func", [torch.add, torch.sub])
def test_binary_with_alpha(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


# test binary with scalar
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn(10), "other": torch.tensor(random.uniform(-10, 10))},
        {"input": torch.randn(10), "other": random.uniform(-10,10)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "func",
    [torch.add, torch.sub, torch.mul, torch.div, torch.remainder],
)
def test_binary_with_other_scalar(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn(10), "other": random.uniform(-10, 10)},
        {"input": torch.randn(30), "other": random.uniform(-10, 10)},
        {"input": torch.randn(30), "other": random.randint(-10, 10)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "func",
    [
        torch.ge,
        torch.gt,
        torch.eq,
        torch.ne,
        torch.greater,
        torch.greater_equal,
        torch.lt,
        torch.le,
    ],
)
def test_binary_compare_with_other_scalar(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


@pytest.mark.parametrize(
    "input_data",
    [
        {"input": random.uniform(-10, 10), "other": torch.randn(10)},
        {"input": torch.tensor(random.uniform(-10, 10)), "other": torch.randn(10)},
        {"input": torch.tensor(random.randint(-10, 10)), "other": torch.randn(10)},
    ],
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize(
    "func",
    [torch.add, torch.sub, torch.mul],
)
def test_binary_with_input_scalar(input_data, dtype, func):
    function(input_data, dtype, dtype, func)
