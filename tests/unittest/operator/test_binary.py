"""Test binary operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, W0106
import random
import numpy as np
import pytest
import torch
from torch_musa import testing

input_datas = [
    {"input": torch.tensor(random.uniform(-10, 10)), "other": torch.randn(30, 30)},
    {"input": torch.randn(30), "other": torch.tensor(random.uniform(-10, 10))},
    {"input": torch.randn(30, 1), "other": torch.randn(30, 30)},
    {"input": torch.randn(30, 1), "other": torch.randn(1, 30)},
]

for data in testing.get_raw_data():
    input_datas.append({"input": data, "other": data})

all_funcs_except_div = [
    torch.atan2,
    torch.add,
    torch.sub,
    torch.mul,
    torch.eq,
    torch.ne,
    torch.gt,
    torch.ge,
    torch.greater_equal,
    torch.greater,
    torch.min,
    torch.max,
    torch.le,
]

all_support_types = testing.get_all_support_types()


def function(input_data, dtype, other_dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    if "other" in input_data.keys() and isinstance(input_data["other"], torch.Tensor):
        input_data["other"] = input_data["other"].to(other_dtype)
    if func in (torch.div, torch.remainder):
        input_data["other"] = abs(input_data["other"])
    if func in (torch.pow,):
        input_data["exponent"] = input_data["exponent"].to(other_dtype)
    test = testing.OpTest(func=func, input_args=input_data)
    test.check_result()


# normal case
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("other_dtype", all_support_types)
@pytest.mark.parametrize("func", all_funcs_except_div)
def test_binary(input_data, dtype, other_dtype, func):
    function(input_data, dtype, other_dtype, func)


# test div, remainder, floor_divide which only support float and make sure other is not zero
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("func", [torch.div, torch.remainder, torch.floor_divide])
def test_div(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


# test add_alpha and sub_alpha(only support float)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {
            "input": torch.randn(30),
            "other": torch.randn(30),
            "alpha": random.uniform(-10, 10),
        },
        {
            "input": torch.randn(30),
            "other": torch.tensor(random.uniform(-100, 100)),
            "alpha": random.uniform(-10, 10),
        },
        {
            "input": random.uniform(0, 1),
            "other": random.uniform(-1, 1),
            "alpha": random.uniform(-1, 1),
        },
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("func", [torch.add, torch.sub])
def test_binary_with_alpha(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


# test binary with scalar
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn(10), "other": torch.tensor(random.uniform(-10, 10))},
        {"input": torch.randn(10), "other": random.uniform(-10, 10)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "func",
    [torch.add, torch.sub, torch.mul, torch.div, torch.remainder],
)
def test_binary_with_other_scalar(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
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


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
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


# torch.bitwise_and/or/xor not support torch.float32
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn(10), "other": torch.randn(10)},
        {"input": torch.randn(10, 10), "other": torch.randn(10, 10)},
        {"input": torch.randn(10, 10, 2), "other": torch.randn(10, 10, 2)},
        {"input": torch.randn(10, 10, 2, 2), "other": torch.randn(10, 10, 2, 2)},
        {
            "input": torch.randn(10, 10, 2, 2, 1),
            "other": torch.randn(10, 10, 2, 2, 1),
        },
        {
            "input": torch.randn(10, 10, 2, 2, 1, 3),
            "other": torch.randn(10, 10, 2, 2, 1, 3),
        },
        {
            "input": torch.randn(10, 10, 2, 2, 1, 3, 2),
            "other": torch.randn(10, 10, 2, 2, 1, 3, 2),
        },
        {
            "input": torch.randn(10, 10, 2, 2, 1, 3, 2, 2),
            "other": torch.randn(10, 10, 2, 2, 1, 3, 2, 2),
        },
        {"input": torch.tensor(1.2), "other": torch.randn(30, 30)},
        {"input": torch.randn(30), "other": torch.tensor(1.2)},
        {"input": torch.randn(30, 1), "other": torch.randn(30, 30)},
        {"input": torch.randn(30, 1), "other": torch.randn(1, 30)},
    ],
)
# TODO(@mingyuan-wang): `torch.bitwise_and(..., dtype=torch.int64)` will fail
# with the new(20230525) musatoolkit, enable `torch.int64` once solved
# @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.int32])
@pytest.mark.parametrize(
    "func",
    [torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor],
)
def test_bitwise(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {
            "input": torch.zeros(
                5,
            ),
            "other": torch.tensor([-1, 0, 1, float("inf"), float("nan")]),
        },
        {"input": torch.randn(10), "other": torch.randn(10)},
        {"input": torch.randn(10, 10), "other": torch.randn(10, 10)},
        {"input": torch.randn(10, 10, 2), "other": torch.randn(10, 10, 2)},
        {"input": torch.randn(10, 10, 2, 2), "other": torch.randn(10, 10, 2, 2)},
        {
            "input": torch.linspace(-50, 50, 2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6),
            "other": torch.linspace(-60, 60, 2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6),
        },
        {
            "input": torch.randint(low=-50, high=50, size=[4, 5, 6, 7, 8, 2]),
            "other": torch.randint(low=-50, high=50, size=[4, 5, 6, 7, 8, 2]),
        },
        {
            "input": torch.randn(6, 6, 3, 4, 5, 2, 2),
            "other": torch.randn(6, 6, 3, 4, 5, 2, 2),
        },
        {
            "input": torch.randn(6, 6, 3, 4, 5, 2, 2, 3),
            "other": torch.randn(6, 6, 3, 4, 5, 2, 2, 3),
        },
        {"input": torch.tensor(3.14159), "other": torch.randn(25, 25)},
        {"input": torch.randn(30), "other": torch.tensor(3.14159)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
@pytest.mark.parametrize(
    "func",
    [torch.xlogy],
)
def test_xlogy(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {
            "input": torch.tensor([True, False, True]),
            "other": torch.tensor([True, False, False]),
        },
        {
            "input": torch.randint(low=0, high=10, size=[2, 2, 3]),
            "other": torch.randint(low=0, high=10, size=[2, 2, 3]),
        },
        {
            "input": torch.randint(low=0, high=10, size=[2, 2, 3, 4]),
            "other": torch.randint(low=0, high=10, size=[2, 2, 3, 4]),
        },
        {
            "input": torch.randint(low=0, high=10, size=[2, 2, 3, 4, 4]),
            "other": torch.randint(low=0, high=10, size=[2, 2, 3, 4, 4]),
        },
        {
            "input": torch.randint(low=0, high=10, size=[4, 5, 6, 7, 8, 2]),
            "other": torch.randint(low=0, high=10, size=[4, 5, 6, 7, 8, 2]),
        },
        {
            "input": torch.randint(low=0, high=10, size=[4, 5, 6, 7, 8, 2, 3]),
            "other": torch.randint(low=0, high=10, size=[4, 5, 6, 7, 8, 2, 3]),
        },
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
@pytest.mark.parametrize(
    "func",
    [torch.logical_and],
)
def test_logical_and(input_data, dtype, func):
    function(input_data, dtype, dtype, func)


pow_input_datas = [
    {"input": torch.randn(60), "exponent": torch.tensor(2.0)},
    {
        "input": torch.randn(60, 2),
        "exponent": torch.randint(low=1, high=10, size=(60, 2)),
    },
    {
        "input": torch.randn(60, 2, 3),
        "exponent": torch.randint(low=1, high=10, size=(60, 2, 3)),
    },
    {
        "input": torch.randn(60, 2, 3, 4),
        "exponent": torch.randint(low=1, high=10, size=(60, 2, 3, 4)),
    },
    {
        "input": torch.arange(0, 24 * 5).reshape(1, 2, 3, 4, -1),
        "exponent": torch.randint(low=1, high=10, size=(1, 2, 3, 4, 5)),
    },
    {
        "input": torch.arange(0, 24 * 5).reshape(1, 2, 3, 4, 5, -1),
        "exponent": torch.randint(low=1, high=10, size=(1, 2, 3, 4, 5, 1)),
    },
    {
        "input": torch.linspace(-10, 10, 24 * 5 * 6).reshape(1, 2, 3, 4, 5, 6, -1),
        "exponent": torch.randint(low=1, high=10, size=(1, 2, 3, 4, 5, 6, 1)),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", pow_input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pow_tensor(input_data, dtype):
    function(input_data, dtype, dtype, torch.pow)


equal_input_datas = [
    {"input": torch.randn(60), "other": torch.randn(60)},
    {"input": 10 * torch.randn(60), "other": 10 * torch.randn(60)},
    {"input": 10 * torch.randn(60, 4), "other": 10 * torch.randn(60, 4)},
    {"input": 5 * torch.randn(60, 4, 6), "other": 5 * torch.randn(60, 4, 6)},
    {"input": 5 * torch.randn(60, 4, 6, 7), "other": 5 * torch.randn(60, 4, 6, 7)},
    {"input": 5 * torch.ones(60, 4, 6, 7), "other": 5 * torch.ones(60, 4, 6, 7)},
    {"input": 5 * torch.zeros(60, 4, 6, 7), "other": 5 * torch.zeros(60, 4, 6, 7)},
    {
        "input": 20 * torch.randn(60, 4, 6, 7, 8),
        "other": 20 * torch.randn(60, 4, 6, 7, 8),
    },
    {"input": torch.randn(60, 2), "other": torch.randint(low=1, high=10, size=(60, 2))},
    {
        "input": torch.randn(60, 2, 3),
        "other": torch.randint(low=1, high=10, size=(60, 2, 3)),
    },
    {
        "input": torch.randn(60, 2, 3, 4),
        "other": torch.randint(low=1, high=10, size=(60, 2, 3, 4)),
    },
    {
        "input": torch.arange(0, 24 * 5).reshape(1, 2, 3, 4, -1),
        "other": torch.randint(low=1, high=10, size=(1, 2, 3, 4, 5)),
    },
    {
        "input": torch.arange(0, 24 * 5).reshape(1, 2, 3, 4, 5, -1),
        "other": torch.randint(low=1, high=10, size=(1, 2, 3, 4, 5, 1)),
    },
    {
        "input": torch.linspace(-10, 10, 24 * 5 * 6).reshape(1, 2, 3, 4, 5, 6, -1),
        "other": torch.randint(low=1, high=10, size=(1, 2, 3, 4, 5, 6, 1)),
    },
    {"input": torch.randn(60) < 0, "other": torch.randn(60) < 0},
    {"input": torch.randn(60, 4) < 0, "other": torch.randn(60, 4) < 0},
    {"input": torch.randn(60, 4, 5) < 0, "other": torch.randn(60, 4, 5) < 0},
    {"input": torch.randn(60, 4, 5, 70) < 0, "other": torch.randn(60, 4, 5, 70) < 0},
] + [
    {"input": data.clone(), "other": data.clone()}
    for data in [10 * torch.randn(1, 2) for _ in range(10)]
]+[
    {"input": data.clone(), "other": data.clone()}
    for data in [10 * torch.randn(1, 2, 3, 4, 10, 5) for _ in range(10)]
]+[
    {"input": data.clone(), "other": data.clone()}
    for data in [10 * torch.randn(1, 2, 3, 4, 10, 20, 5) for _ in range(10)]
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", equal_input_datas)
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool]
)
@pytest.mark.parametrize(
    "other_dtype", [torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool]
)
def test_equal_tensor(input_data, dtype, other_dtype):
    function(input_data, dtype, other_dtype, torch.equal)
