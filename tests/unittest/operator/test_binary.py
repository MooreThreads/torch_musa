"""Test binary operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, W0106
import random
import copy
import pytest
import torch
import torch_musa
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
# bf16 is not supported on arch older than qy2
if testing.get_musa_arch() >= 22:
    all_support_types.append(torch.bfloat16)


def function(input_data_, dtype, other_dtype, func):
    input_data = copy.deepcopy(input_data_)
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    if "other" in input_data.keys() and isinstance(input_data["other"], torch.Tensor):
        input_data["other"] = input_data["other"].to(other_dtype)
    if func in (torch.div, torch.remainder):
        input_data["other"] = abs(input_data["other"])
    if func in (torch.pow,):
        input_data["exponent"] = input_data["exponent"].to(other_dtype)
    comparator = testing.DefaultComparator(equal_nan=True)
    if torch.bfloat16 in (dtype, other_dtype):
        comparator = testing.DefaultComparator(
            abs_diff=1, rel_diff=1e-1, equal_nan=True
        )
    if torch.float16 in (dtype, other_dtype):
        comparator = testing.DefaultComparator(
            abs_diff=1e-3, rel_diff=1e-3, equal_nan=True
        )
    test = testing.OpTest(func=func, input_args=input_data, comparators=comparator)
    # test = testing.OpTest(func=func, input_args=input_data)
    if torch.float16 in (dtype, other_dtype):
        test.check_musafp16_vs_musafp32()
        test.check_out_ops(fp16=True)
        test.check_grad_fn(fp16=True)
    elif torch.bfloat16 in (dtype, other_dtype):
        test.check_musabf16_vs_musafp16()
        test.check_out_ops(bf16=True)
        test.check_grad_fn(bf16=True)
    else:
        test.check_result()
        test.check_out_ops()
        test.check_grad_fn()


# normal case
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("other_dtype", all_support_types)
@pytest.mark.parametrize("func", all_funcs_except_div)
def test_binary(input_data, dtype, other_dtype, func):
    function(input_data, dtype, other_dtype, func)


all_funcs_inplace = [
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
    torch.le,
    torch.div,
    torch.remainder,
    torch.floor_divide,
]
inplace_data = (
    [{"input": data, "other": torch.rand_like(data)} for data in testing.get_raw_data()]
    + [
        {"input": data, "other": torch.tensor(random.random())}
        for data in testing.get_raw_data()
    ]
    + [{"input": data, "other": random.random()} for data in testing.get_raw_data()]
)
inplace_dtype = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    inplace_dtype.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inplace_data)
@pytest.mark.parametrize("func", all_funcs_inplace)
@pytest.mark.parametrize("dtype", inplace_dtype)
def test_binary_inplace_addr_and_value(input_data, func, dtype):
    inp_data = input_data["input"].clone()
    if isinstance(input_data["other"], torch.Tensor):
        other = input_data["other"].clone().to(dtype)
    else:
        other = copy.deepcopy(input_data["other"])
    if func == torch.atan2 and not isinstance(other, torch.Tensor):
        other = torch.tensor(other).to(dtype)
    abs_diff, rel_diff = 5e-3, 5e-4
    if dtype in [torch.bfloat16, torch.float16]:
        abs_diff, rel_diff = 5e-2, 5e-3
    # The torch.floor_divide function is a combination of
    # division and floor operations, and therefore it may introduce larger errors.
    if func == torch.floor_divide:
        abs_diff, rel_diff = 1, 1e-1
        inp_data *= 10.0
        other *= 10.0
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=inp_data,
        input_args={"other": other},
        comparators=[
            testing.DefaultComparator(
                abs_diff=abs_diff, rel_diff=rel_diff, equal_nan=True
            )
        ],
    )
    test.check_address()
    if func == torch.atan2:
        test.check_res(cpu_to_fp32=True)
    else:
        test.check_res()


# test div, remainder, floor_divide which only support float and make sure other is not zero
dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


input_datas_for_div = [
    {
        "input": torch.tensor(random.uniform(-10, 10)) * 100,
        "other": torch.rand(30, 30).uniform_(1, 2),
    },
    {
        "input": torch.rand(30).uniform_(-2, 2) * 100,
        "other": torch.tensor(random.uniform(1, 2)),
    },
    {
        "input": torch.rand(30, 1).uniform_(-2, 2) * 100,
        "other": torch.randn(30, 30).uniform_(1, 2),
    },
    {
        "input": torch.rand(30, 1).uniform_(-2, 2) * 100,
        "other": torch.randn(1, 30).uniform_(1, 2),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas_for_div)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("func", [torch.div, torch.remainder, torch.floor_divide])
def test_div(input_data, dtype, func):
    if func == torch.floor_divide and dtype == torch.bfloat16:
        test = testing.OpTest(
            func=func,
            input_args=input_data,
            comparators=testing.DefaultComparator(
                abs_diff=1, rel_diff=1e-1, equal_nan=True
            ),
        )
        test.check_musabf16_vs_musafp16()
        test.check_out_ops(bf16=True)
        test.check_grad_fn(bf16=True)
    elif func == torch.div and dtype == torch.float16:
        test = testing.OpTest(
            func=func,
            input_args=input_data,
            comparators=testing.DefaultComparator(
                abs_diff=1e-1, rel_diff=1e-3, equal_nan=True
            ),
        )
        test.check_musafp16_vs_musafp32()
        test.check_out_ops(fp16=True)
        test.check_grad_fn(fp16=True)
    else:
        function(input_data, dtype, dtype, func)


# test add_alpha and sub_alpha(only support float)
dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


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
            "other": torch.tensor(random.uniform(-5, 5)),
            "alpha": random.uniform(-5, 5),
        },
        {
            "input": random.uniform(0, 1),
            "other": random.uniform(-1, 1),
            "alpha": random.uniform(-1, 1),
        },
        {
            "input": torch.tensor(random.uniform(0, 1)),
            "other": random.uniform(-1, 1),
            "alpha": random.uniform(-1, 1),
        },
        {
            "input": random.uniform(0, 1),
            "other": torch.tensor(random.uniform(-1, 1)),
            "alpha": random.uniform(-1, 1),
        },
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("func", [torch.add, torch.sub])
def test_binary_with_alpha(input_data, dtype, func):
    function(input_data, dtype, dtype, func)
    input_data["input"] = torch.tensor(input_data["input"])
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=input_data["input"].to(dtype),
        input_args={"other": input_data["other"], "alpha": input_data["alpha"]},
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res()


# test binary with scalar
dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn(10), "other": torch.tensor(random.uniform(-10, 10))},
        {"input": torch.randn(10), "other": random.uniform(-10, 10)},
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "func",
    [torch.add, torch.sub, torch.mul, torch.div, torch.remainder],
)
def test_binary_with_other_scalar(input_data, dtype, func):
    function(input_data, dtype, dtype, func)
    if not isinstance(input_data["other"], torch.Tensor):
        input_data["other"] = torch.tensor(input_data["other"])
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=input_data["input"].to(dtype),
        input_args={"other": input_data["other"].to(dtype)},
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res()


dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn(10), "other": random.uniform(-10, 10)},
        {"input": torch.randn(30), "other": random.uniform(-10, 10)},
        {"input": torch.randn(30), "other": random.randint(-10, 10)},
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
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
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=input_data["input"].to(dtype),
        input_args={"other": input_data["other"]},
    )
    test.check_address()
    test.check_res()


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
        {"input": torch.randn(10, 10, 0), "other": torch.randn(10, 10, 0)},
        {"input": torch.randn(0, 10, 2), "other": torch.randn(0, 10, 2)},
        {"input": torch.randn(10, 10, 2, 2), "other": torch.randn(10, 10, 2, 2)},
        {
            "input": torch.randn(10, 11, 1, 1).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 11, 1, 1),
        },
        {
            "input": torch.randn(10, 1, 2, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 2, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 1, 1, 1).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 1, 1),
        },
        {
            "input": torch.randn(10, 1, 1, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 1, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 7, 5, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 7, 5, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 7, 0, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 7, 0, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 0, 5, 0).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 0, 5, 0).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 7, 5, 3),
            "other": torch.randn(10, 7, 5, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 7, 5, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 7, 5, 3),
        },
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


# torch.bitwise_and/or/xor not support torch.float32
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn(10), "other": torch.randn(10)},
        {"input": torch.randn(10, 10), "other": torch.randn(10, 10)},
        {"input": torch.randn(10, 10, 2), "other": torch.randn(10, 10, 2)},
        {"input": torch.randn(10, 10, 0), "other": torch.randn(10, 10, 0)},
        {"input": torch.randn(0, 10, 2), "other": torch.randn(0, 10, 2)},
        {"input": torch.randn(10, 10, 2, 2), "other": torch.randn(10, 10, 2, 2)},
        {
            "input": torch.randn(10, 11, 1, 1).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 11, 1, 1),
        },
        {
            "input": torch.randn(10, 1, 2, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 2, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 1, 1, 1).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 1, 1),
        },
        {
            "input": torch.randn(10, 1, 1, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 1, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 7, 5, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 7, 5, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 7, 0, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 7, 0, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 0, 5, 0).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 0, 5, 0).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 7, 5, 3),
            "other": torch.randn(10, 7, 5, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 7, 5, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 7, 5, 3),
        },
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
        {"input": torch.randn(30), "other": torch.tensor(1.2)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.int32])
@pytest.mark.parametrize(
    "func",
    [torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor],
)
def test_bitwise_inplace_address(input_data, dtype, func):
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=input_data["input"].to(dtype),
        input_args={"other": input_data["other"].to(dtype)},
    )
    test.check_address()
    test.check_res()


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
        {"input": torch.randn(10, 10, 0, 0), "other": torch.randn(10, 10, 0, 0)},
        {"input": torch.randn(10, 0, 2, 2), "other": torch.randn(10, 0, 2, 2)},
        {"input": torch.randn(0, 0, 2, 2), "other": torch.randn(0, 0, 2, 2)},
        {
            "input": torch.randn(10, 11, 1, 1).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 11, 1, 1),
        },
        {
            "input": torch.randn(10, 1, 2, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 2, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 1, 0, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 0, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(0, 1, 2, 0).to(memory_format=torch.channels_last),
            "other": torch.randn(0, 1, 2, 0).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 1, 1, 1).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 1, 1),
        },
        {
            "input": torch.randn(10, 1, 1, 3),
            "other": torch.randn(10, 1, 1, 3).to(memory_format=torch.channels_last),
        },
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
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.int32, torch.int64]
)
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
            "input": torch.zeros(
                5,
            ),
            "other": torch.tensor([-1, 0, 1, float("inf"), float("nan")]),
        },
        {"input": torch.randn(10), "other": torch.randn(10)},
        {"input": torch.randn(10, 10), "other": torch.randn(10, 10)},
        {"input": torch.randn(10, 10, 2), "other": torch.randn(10, 10, 2)},
        {"input": torch.randn(10, 10, 0, 0), "other": torch.randn(10, 10, 0, 0)},
        {"input": torch.randn(10, 0, 2, 2), "other": torch.randn(10, 0, 2, 2)},
        {"input": torch.randn(0, 0, 2, 2), "other": torch.randn(0, 0, 2, 2)},
        {"input": torch.randn(10, 10, 2, 2), "other": torch.randn(10, 10, 2, 2)},
        {
            "input": torch.randn(10, 11, 1, 1).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 11, 1, 1),
        },
        {
            "input": torch.randn(10, 1, 2, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 2, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 1, 0, 3).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 0, 3).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(0, 1, 2, 0).to(memory_format=torch.channels_last),
            "other": torch.randn(0, 1, 2, 0).to(memory_format=torch.channels_last),
        },
        {
            "input": torch.randn(10, 1, 1, 1).to(memory_format=torch.channels_last),
            "other": torch.randn(10, 1, 1, 1),
        },
        {
            "input": torch.randn(10, 1, 1, 3),
            "other": torch.randn(10, 1, 1, 3).to(memory_format=torch.channels_last),
        },
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
        {"input": torch.randn(30), "other": torch.tensor(3.14159)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "func",
    [torch.xlogy],
)
def test_xlogy_inplace(input_data, dtype, func):
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=input_data["input"].to(dtype),
        input_args={"other": input_data["other"].to(dtype)},
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res()


dtypes = [torch.float32, torch.float16, torch.int32, torch.int64]
if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


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
            "input": torch.randint(low=0, high=10, size=[2, 2, 1, 1]),
            "other": torch.randint(low=0, high=10, size=[2, 2, 1, 1]).to(
                memory_format=torch.channels_last
            ),
        },
        {
            "input": torch.randint(low=0, high=10, size=[2, 1, 3, 4]),
            "other": torch.randint(low=0, high=10, size=[2, 1, 3, 4]).to(
                memory_format=torch.channels_last
            ),
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
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "func",
    [torch.logical_and, torch.logical_or, torch.logical_xor],
)
def test_logical_(input_data, dtype, func):
    function(input_data, dtype, dtype, func)
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=input_data["input"].to(dtype),
        input_args={"other": input_data["other"].to(dtype)},
    )
    test.check_address()
    test.check_res()


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
        "input": torch.randn(0, 2, 3),
        "exponent": torch.randint(low=1, high=10, size=(0, 2, 3)),
    },
    {
        "input": torch.randn(60, 0, 0),
        "exponent": torch.randint(low=1, high=10, size=(60, 0, 0)),
    },
    {
        "input": torch.randn(60, 2, 3, 4),
        "exponent": torch.randint(low=1, high=10, size=(60, 2, 3, 4)),
    },
    {
        "input": torch.randn(60, 1, 3, 4).to(memory_format=torch.channels_last),
        "exponent": torch.randint(low=1, high=10, size=(60, 1, 3, 4)),
    },
    {
        "input": torch.randn(60, 0, 3, 4).to(memory_format=torch.channels_last),
        "exponent": torch.randint(low=1, high=10, size=(60, 0, 3, 4)),
    },
    {
        "input": torch.randn(0, 1, 3, 0).to(memory_format=torch.channels_last),
        "exponent": torch.randint(low=1, high=10, size=(0, 1, 3, 0)),
    },
    {
        "input": torch.randn(60, 2, 1, 1).to(memory_format=torch.channels_last),
        "exponent": torch.randint(low=1, high=10, size=(60, 2, 1, 1)),
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
    test = testing.InplaceOpChek(
        func_name=torch.pow.__name__ + "_",
        self_tensor=input_data["input"].to(dtype),
        input_args={"exponent": input_data["exponent"].to(dtype)},
    )
    test.check_address()
    test.check_res()


equal_input_datas = (
    [
        {"input": torch.randn(60), "other": torch.randn(60)},
        {"input": 10 * torch.randn(60), "other": 10 * torch.randn(60)},
        {"input": 10 * torch.randn(60, 4), "other": 10 * torch.randn(60, 4)},
        {"input": 5 * torch.randn(60, 4, 6), "other": 5 * torch.randn(60, 4, 6)},
        {"input": 5 * torch.randn(60, 4, 6, 7), "other": 5 * torch.randn(60, 4, 6, 7)},
        {"input": 5 * torch.randn(60, 4, 6, 0), "other": 5 * torch.randn(60, 4, 6, 0)},
        {"input": 5 * torch.randn(0, 4, 6, 5), "other": 5 * torch.randn(0, 4, 6, 5)},
        {"input": 5 * torch.randn(60, 4, 0, 0), "other": 5 * torch.randn(60, 4, 0, 0)},
        {"input": 5 * torch.ones(60, 4, 6, 7), "other": 5 * torch.ones(60, 4, 6, 7)},
        {
            "input": 5 * torch.ones(60, 1, 6, 7).to(memory_format=torch.channels_last),
            "other": 5 * torch.ones(60, 1, 6, 7).to(memory_format=torch.channels_last),
        },
        {
            "input": 5 * torch.ones(60, 2, 1, 1).to(memory_format=torch.channels_last),
            "other": 5 * torch.ones(60, 2, 1, 1).to(memory_format=torch.channels_last),
        },
        {
            "input": 5 * torch.ones(60, 0, 1, 1).to(memory_format=torch.channels_last),
            "other": 5 * torch.ones(60, 0, 1, 1).to(memory_format=torch.channels_last),
        },
        {
            "input": 5 * torch.ones(60, 2, 1, 0).to(memory_format=torch.channels_last),
            "other": 5 * torch.ones(60, 2, 1, 0).to(memory_format=torch.channels_last),
        },
        {
            "input": 5 * torch.ones(0, 2, 1, 1).to(memory_format=torch.channels_last),
            "other": 5 * torch.ones(0, 2, 1, 1).to(memory_format=torch.channels_last),
        },
        {"input": 5 * torch.zeros(60, 4, 6, 7), "other": 5 * torch.zeros(60, 4, 6, 7)},
        {
            "input": 20 * torch.randn(60, 4, 6, 7, 8),
            "other": 20 * torch.randn(60, 4, 6, 7, 8),
        },
        {
            "input": torch.randn(60, 2),
            "other": torch.randint(low=1, high=10, size=(60, 2)),
        },
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
        {
            "input": torch.randn(60, 4, 5, 70) < 0,
            "other": torch.randn(60, 4, 5, 70) < 0,
        },
    ]
    + [
        {"input": data.clone(), "other": data.clone()}
        for data in [10 * torch.randn(1, 2) for _ in range(10)]
    ]
    + [
        {"input": data.clone(), "other": data.clone()}
        for data in [10 * torch.randn(1, 2, 3, 4, 10, 5) for _ in range(10)]
    ]
    + [
        {"input": data.clone(), "other": data.clone()}
        for data in [10 * torch.randn(1, 2, 3, 4, 10, 20, 5) for _ in range(10)]
    ]
)


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


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {
            "lhs": torch.randint(low=1, high=6, size=(2, 3)),
            "rhs": 2,
            "func": lambda l, r: torch.div(l, r, rounding_mode="floor"),
            "reverse": True,
        },
        {
            "lhs": torch.randn(8).to(torch.uint8),
            "rhs": 255.0,
            "func": lambda l, r: l / r,
            "reverse": False,
        },
        {
            "lhs": torch.randn(1024),
            "rhs": torch.tensor(15).long(),
            "func": lambda l, r: l / r,
            "reverse": True,
        },
    ],
)
def test_rc120_binary_issues(input_data):
    l_cpu, r_cpu = input_data["lhs"], input_data["rhs"]
    func = input_data["func"]
    reverse = input_data["reverse"]

    def to_musa(c):
        if isinstance(c, torch.Tensor):
            return c.musa()
        return c

    l_musa, r_musa = to_musa(l_cpu), to_musa(r_cpu)
    cmp = testing.DefaultComparator(equal_nan=True)

    o_cpu = func(l_cpu, r_cpu)
    o_musa = func(l_musa, r_musa)
    assert cmp(o_cpu, o_musa.cpu())

    if reverse:
        o_cpu_rev = func(r_cpu, l_cpu)
        o_musa_rev = func(r_musa, l_musa)
        assert cmp(o_cpu_rev, o_musa_rev.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {
            "lhs": torch.randint(low=0, high=10, size=(2, 3)),
            "rhs": 5,
            "reverse": True,
        },
        {
            "lhs": torch.randint(low=0, high=10, size=(2, 3)),
            "rhs": torch.tensor([5]),
            "reverse": True,
        },
    ],
)
@pytest.mark.parametrize("func", [torch.add, torch.sub])
@pytest.mark.parametrize("dtype", [torch.uint8])
def test_uint_binary_add_sub(input_data, func, dtype):
    def to_int_type(c, d_t):
        if isinstance(c, torch.Tensor):
            return c.to(d_t)
        return int(c)

    l_cpu, r_cpu = input_data["lhs"], input_data["rhs"]
    l_cpu, r_cpu = to_int_type(l_cpu, dtype), to_int_type(r_cpu, dtype)
    reverse = input_data["reverse"]

    def to_musa(c):
        if isinstance(c, torch.Tensor):
            return c.musa()
        return c

    l_musa, r_musa = to_musa(l_cpu), to_musa(r_cpu)
    cmp = testing.DefaultComparator(equal_nan=True)

    def test(c_l, c_r, m_l, m_r, f):
        o_c = f(c_l, c_r)
        o_m = f(m_l, m_r)
        assert o_c.dtype == o_m.dtype
        assert cmp(o_c, o_m.cpu())

    test(l_cpu, r_cpu, l_musa, r_musa, func)
    if reverse:
        test(r_cpu, l_cpu, r_musa, l_musa, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_boolean_binary_add():
    c_x, c_y = torch.randn((128, 128)), torch.randn((128, 128))
    m_x, m_y = c_x.musa(), c_y.musa()

    c_x, c_y = c_x > 0.5, c_y > 0.5
    m_x, m_y = m_x > 0.5, m_y > 0.5

    c_x += c_y
    m_x += m_y

    assert c_x.dtype == torch.bool and c_x.dtype == m_x.dtype
    cmp = testing.BooleanComparator()
    assert cmp(c_x, m_x.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "shape",
    [
        {
            "shape": [128, 128],
            "uncontig_func": lambda t: t[:64, 64:],
        },
    ],
)
@pytest.mark.parametrize(
    "io_types",
    [
        [[torch.uint8, torch.int8], [torch.float16, torch.int32, torch.float]],
        [[torch.int16], [torch.int32, torch.float]],
        [[torch.int32], [torch.int64]],
        [[torch.int64], []],
    ],
)
def test_binary_fmod_int(shape, io_types):
    i_raw = torch.randint(-100, 100, shape["shape"])
    a_raw = torch.randint(-100, 100, shape["shape"])
    a_raw[::2, :] = 2
    a_raw[1::2, :] = -2
    i_ts, o_ts = io_types
    u_f = shape["uncontig_func"]

    def do_assert(golden, result):
        assert golden.dtype == result.dtype
        assert golden.shape == result.shape
        assert torch.allclose(golden, result.cpu())

    for i_t in i_ts:
        cpu_i = i_raw.to(i_t)
        musa_i = cpu_i.musa()

        cpu_a = a_raw.to(i_t)
        musa_a = cpu_a.musa()

        cpu_o = cpu_i.fmod(cpu_a)
        musa_o = musa_i.fmod(musa_a)
        do_assert(cpu_o, musa_o)

        uncontig_cpu_o = u_f(cpu_i).fmod(u_f(cpu_a))
        uncontig_musa_o = u_f(musa_i).fmod(u_f(musa_a))
        do_assert(uncontig_cpu_o, uncontig_musa_o)

        cpu_o.copy_(cpu_i)
        cpu_o.fmod_(cpu_a)
        musa_o.copy_(musa_i)
        musa_o.fmod_(musa_a)
        do_assert(cpu_o, musa_o)

        for o_t in o_ts:
            cpu_o.zero_()
            cpu_o = cpu_o.to(o_t)
            torch.fmod(cpu_i, cpu_a, out=cpu_o)
            musa_o.zero_()
            musa_o = musa_o.to(o_t)
            torch.fmod(musa_i, musa_a, out=musa_o)
            do_assert(cpu_o, musa_o)


dtypes = [
    [[torch.float16], [torch.float]],
    [[torch.float], []],
]
if testing.get_musa_arch() >= 22:
    dtypes.append([[torch.bfloat16], [torch.float]])


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("shape", [[128, 128]])
@pytest.mark.parametrize("io_types", dtypes)
def test_binary_fmod_float(shape, io_types):
    i_raw = torch.empty(shape)
    i_raw.uniform_(-9.0, 9.0)
    a_raw = torch.empty(shape)
    a_raw[::3, :] = 2
    a_raw[1::3, :] = -2
    a_raw[2::3, :] = 0
    i_ts, o_ts = io_types

    def do_assert(golden, result):
        if result.dtype == torch.bfloat16:
            assert golden.dtype == result.dtype
        assert golden.shape == result.shape
        atol = 2e-5
        if result.dtype == torch.float16:
            atol = 2e-3
        elif result.dtype == torch.bfloat16:
            atol = 2e-2
        assert torch.allclose(
            golden.float(), result.cpu().float(), atol=atol, equal_nan=True
        )

    for i_t in i_ts:
        cpu_i = i_raw.to(i_t)
        musa_i = cpu_i.musa()

        cpu_a = a_raw.to(i_t)
        musa_a = cpu_a.musa()

        if i_t != torch.bfloat16:
            cpu_i = cpu_i.float()
            cpu_a = cpu_a.float()

        cpu_o = cpu_i.fmod(cpu_a)
        musa_o = musa_i.fmod(musa_a)
        do_assert(cpu_o, musa_o)

        musa_o.copy_(musa_i)
        musa_o.fmod_(musa_a)
        do_assert(cpu_o, musa_o)

        for o_t in o_ts:
            musa_o.zero_()
            musa_o = musa_o.to(o_t)
            torch.fmod(musa_i, musa_a, out=musa_o)
            do_assert(cpu_o, musa_o)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "shape",
    [
        {
            "shape": [128, 128],
            "uncontig_func": lambda t: t[:64, 64:],
        },
    ],
)
@pytest.mark.parametrize(
    "io_types",
    [
        [[torch.uint8, torch.int8], [torch.float16, torch.int32, torch.float]],
        [[torch.int16], [torch.int32, torch.float]],
        [[torch.int32], [torch.int64]],
        [[torch.int64], []],
    ],
)
def test_binary_fmin_fmax_int(shape, io_types):
    i_raw = torch.randint(-100, 100, shape["shape"])
    a_raw = torch.randint(-100, 100, shape["shape"])
    a_raw[::2, :] = 2
    a_raw[1::2, :] = -2
    i_ts, o_ts = io_types
    u_f = shape["uncontig_func"]

    def assert_detail(golden, result):
        assert golden.dtype == result.dtype
        assert golden.shape == result.shape
        assert torch.allclose(golden, result.cpu())

    for i_t in i_ts:
        cpu_i = i_raw.to(i_t)
        musa_i = cpu_i.musa()

        cpu_a = a_raw.to(i_t)
        musa_a = cpu_a.musa()

        cpu_o = cpu_i.fmin(cpu_a)
        musa_o = musa_i.fmin(musa_a)
        assert_detail(cpu_o, musa_o)

        m_cpu_o = cpu_i.fmax(cpu_a)
        m_musa_o = musa_i.fmax(musa_a)
        assert_detail(m_cpu_o, m_musa_o)

        uncontig_cpu_o = u_f(cpu_i).fmin(u_f(cpu_a))
        uncontig_musa_o = u_f(musa_i).fmin(u_f(musa_a))
        assert_detail(uncontig_cpu_o, uncontig_musa_o)

        m_uncontig_cpu_o = u_f(cpu_i).fmax(u_f(cpu_a))
        m_uncontig_musa_o = u_f(musa_i).fmax(u_f(musa_a))
        assert_detail(m_uncontig_cpu_o, m_uncontig_musa_o)

        for o_t in o_ts:
            cpu_o.zero_()
            cpu_o = cpu_o.to(o_t)
            torch.fmin(cpu_i, cpu_a, out=cpu_o)
            musa_o.zero_()
            musa_o = musa_o.to(o_t)
            torch.fmin(musa_i, musa_a, out=musa_o)
            assert_detail(cpu_o, musa_o)

        for o_t in o_ts:
            cpu_o.zero_()
            cpu_o = cpu_o.to(o_t)
            torch.fmax(cpu_i, cpu_a, out=cpu_o)
            musa_o.zero_()
            musa_o = musa_o.to(o_t)
            torch.fmax(musa_i, musa_a, out=musa_o)
            assert_detail(cpu_o, musa_o)


dtypes = [
    [[torch.float16], [torch.float]],
    [[torch.float], []],
]
if testing.get_musa_arch() >= 22:
    dtypes.append([[torch.bfloat16], [torch.float]])


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("shape", [[128, 128]])
@pytest.mark.parametrize("io_types", dtypes)
def test_binary_fmin_fmax_float(shape, io_types):
    i_raw = torch.empty(shape)
    i_raw.uniform_(-9.0, 9.0)
    a_raw = torch.empty(shape)
    a_raw[::3, :] = 2
    a_raw[1::3, :] = -2
    a_raw[2::3, :] = 0
    i_ts, o_ts = io_types

    def assert_detail(golden, result):
        if result.dtype == torch.bfloat16:
            assert golden.dtype == result.dtype
        assert golden.shape == result.shape
        atol = 2e-5
        if result.dtype == torch.float16:
            atol = 2e-3
        elif result.dtype == torch.bfloat16:
            atol = 2e-2
        assert torch.allclose(
            golden.float(), result.cpu().float(), atol=atol, equal_nan=True
        )

    for i_t in i_ts:
        cpu_i = i_raw.to(i_t)
        musa_i = cpu_i.musa()

        cpu_a = a_raw.to(i_t)
        musa_a = cpu_a.musa()

        if i_t != torch.bfloat16:
            cpu_i = cpu_i.float()
            cpu_a = cpu_a.float()

        cpu_o = cpu_i.fmin(cpu_a)
        musa_o = musa_i.fmin(musa_a)
        assert_detail(cpu_o, musa_o)

        musa_o.copy_(musa_i)
        musa_o = musa_o.fmin(musa_a)
        assert_detail(cpu_o, musa_o)

        m_cpu_o = cpu_i.fmax(cpu_a)
        m_musa_o = musa_i.fmax(musa_a)
        assert_detail(m_cpu_o, m_musa_o)

        m_musa_o.copy_(musa_i)
        m_musa_o = m_musa_o.fmax(musa_a)
        assert_detail(m_cpu_o, m_musa_o)

        for o_t in o_ts:
            musa_o.zero_()
            musa_o = musa_o.to(o_t)
            torch.fmin(musa_i, musa_a, out=musa_o)
            assert_detail(cpu_o, musa_o)

        for o_t in o_ts:
            m_musa_o.zero_()
            m_musa_o = m_musa_o.to(o_t)
            torch.fmax(musa_i, musa_a, out=m_musa_o)
            assert_detail(m_cpu_o, m_musa_o)
