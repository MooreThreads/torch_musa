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

input_datas = [
    {"input": torch.randn(60)},
    {"input": torch.randn(60, 1)},
]
float_dtypes = [torch.float32, torch.float16]
all_dtypes = testing.get_all_support_types()
# bf16 is not supported on arch older than qy2
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)
    all_dtypes.append(torch.bfloat16)


for data in testing.get_raw_data():
    input_datas.append({"input": data})

all_basic_funcs = [
    torch.abs,
    torch.sqrt,
    torch.rsqrt,
    torch.tanh,
    torch.tan,
    torch.reciprocal,
    torch.sigmoid,
    torch.exp,
    torch.cos,
    torch.sin,
    torch.log,
    torch.acos,
    torch.atan,
    torch.round,
    torch.sgn,
    torch.log10,
    torch.log2,
    torch.floor,
    torch.logical_not,
]

all_inplace_funcs = [
    torch.abs_,
    torch.sqrt_,
    torch.rsqrt_,
    torch.tan_,
    torch.tanh_,
    torch.reciprocal_,
    torch.sigmoid_,
    torch.exp_,
    torch.cos_,
    torch.sin_,
    torch.log_,
    torch.acos_,
    torch.atan_,
    torch.round_,
    torch.log10_,
    torch.floor_,
]


all_nn_funcs = [
    torch.nn.ReLU(),
    torch.nn.GELU(approximate="none"),
    torch.nn.GELU(approximate="tanh"),
    torch.nn.SiLU(),
    torch.nn.LeakyReLU(),
    torch.nn.Hardswish(),
    torch.nn.Hardsigmoid(),
]


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
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("func", all_basic_funcs + all_inplace_funcs)
def test_all_basic_funcs(input_data, dtype, func):
    function(input_data, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("func", all_basic_funcs)
def test_all_basic_funcs_out(input_data, dtype, func):
    out = torch.tensor(np.array([]))
    input_args = {"input": input_data["input"], "out": out}
    function(input_args, dtype, func)
    inplace_input = copy.deepcopy(input_data["input"])
    # Prevent inf caused by input absolute values being too small
    if func == torch.reciprocal:
        inplace_input = inplace_input * 100
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_",
        self_tensor=inplace_input.to(dtype),
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
# torch.bitwise_not only support integral and Boolean type
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("func", [torch.bitwise_not])
def test_bitwise_not(input_data, dtype, func):
    function(input_data, dtype, func)
    test = testing.InplaceOpChek(
        func_name=func.__name__ + "_", self_tensor=input_data["input"].to(dtype)
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("func", [torch.bitwise_not])
def test_bitwise_not_out(input_data, dtype, func):
    out = torch.tensor(np.array([]))
    input_args = {"input": input_data["input"], "out": out}
    function(input_args, dtype, func)


# =================================== Test torch.neg begin =================================== #
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_dtypes)
def test_neg(input_data, dtype):
    function(input_data, dtype, torch.neg)
    test = testing.InplaceOpChek(
        func_name=torch.neg.__name__ + "_", self_tensor=input_data["input"].to(dtype)
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_dtypes)
def test_neg_out(input_data, dtype):
    out = torch.tensor(np.array([]))
    input_args = {"input": input_data["input"], "out": out}
    function(input_args, dtype, torch.neg)


# ============================ Test torch.nn.Threshold backward begin ========================== #

threshold_values = [round(random.uniform(-10, 10), 3) for i in range(2)]
to_set_values = [round(random.uniform(-10, 10), 3) for i in range(2)]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("threshold", threshold_values)
@pytest.mark.parametrize("value", to_set_values)
@pytest.mark.parametrize("train", [False, True])
def test_threshold_backward_and_forward(input_data, dtype, threshold, value, train):
    func = torch.nn.Threshold(threshold=threshold, value=value)
    if train:
        comparator = testing.DefaultComparator(abs_diff=1e-6, equal_nan=True)
        tensor = input_data["input"]
        if tensor.requires_grad:
            tensor.grad = None  # reset the grad
        else:
            tensor.requires_grad = True
        test = testing.OpTest(
            func=func,
            input_args=input_data,
            comparators=comparator,
        )
        test.check_result(train=True)
        tensor.requires_grad = False
        test = testing.OpTest(
            func=func,
            input_args=input_data,
            comparators=comparator,
        )
        test.check_grad_fn()
    else:
        function(input_data, dtype, func)


# ============================ Test torch.nn.Threshold backward end ============================= #

# =================================== Test torch.neg end =================================== #


# =================================== Test nn functions begin =================================== #
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("func", all_nn_funcs)
def test_nn_funcs(input_data, dtype, func):
    function(input_data, dtype, func)


# =================================== Test nn functions end ================================== #

# =================================== Test torch.pow begin =================================== #
input_datas = [
    {"input": torch.randn(60), "exponent": 1.0},
    {"input": torch.rand(60, 2), "exponent": 2.0},
    {"input": torch.rand(60, 2, 3), "exponent": 3.0},
    {"input": torch.rand(60, 2, 0), "exponent": 3.0},
    {"input": torch.rand(60, 2, 3, 4), "exponent": 4.0},
    {"input": torch.rand(60, 0, 3, 0), "exponent": 4.0},
    {
        "input": torch.rand(60, 2, 3, 4).to(memory_format=torch.channels_last),
        "exponent": 4.0,
    },
    {
        "input": torch.rand(60, 1, 3, 4).to(memory_format=torch.channels_last),
        "exponent": 4.0,
    },
    {
        "input": torch.rand(60, 1, 0, 4).to(memory_format=torch.channels_last),
        "exponent": 4.0,
    },
    {
        "input": torch.rand(3, 2, 1, 1).to(memory_format=torch.channels_last),
        "exponent": 4.0,
    },
    {"input": torch.arange(0, 24 * 5).reshape(1, 2, 3, 4, -1), "exponent": 2.0},
    {"input": torch.arange(0, 24 * 5).reshape(1, 2, 3, 4, 5, -1), "exponent": 1.0},
    {
        "input": torch.linspace(-10, 10, 24 * 5 * 6).reshape(1, 2, 3, 4, 5, 6, -1),
        "exponent": 8.0,
    },
    {
        "input": torch.linspace(-100, 100, 24 * 5 * 6 * 4).reshape(
            1, 2, 3, 4, 5, 6, 2, 2
        ),
        "exponent": 4.0,
    },
    {
        "input": torch.rand(4, 6, 0, 4, 1),
        "exponent": 4.0,
    },
    {
        "input": torch.rand(4, 6, 0, 4).to(memory_format=torch.channels_last),
        "exponent": 4.0,
    },
    {
        "input": torch.rand(4, 6, 0, 4).to(memory_format=torch.channels_last),
        "exponent": 0.0,
    },
]


# torch.pow only support float32
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_pow(input_data, dtype):
    function(input_data, dtype, torch.pow)
    test = testing.InplaceOpChek(
        func_name=torch.pow.__name__ + "_",
        self_tensor=input_data["input"].to(dtype),
        input_args={"exponent": input_data["exponent"]},
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pow_out(input_data, dtype):
    out = torch.tensor(np.array([]))
    input_args = {"out": out}
    input_args.update(input_data)
    function(input_args, dtype, torch.pow)


# =================================== Test torch.pow end =================================== #


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        torch.randn(2, requires_grad=True),
        torch.randn(2, 0, 16, 0, requires_grad=True),
        torch.randn(2, 3, 16, 16, requires_grad=True),
        torch.randn(2, 3, 0, 16, requires_grad=True),
        torch.randn(2, 3, 16, 16, requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        torch.randn(2, 3, 1, 1, requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        torch.randn(4, 1, 3, 2, requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        torch.randn(4, 1, 1, 1, requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        torch.randn(4, 1, 0, 1, requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        torch.randn(0, 0, 1, 1, requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        torch.randn(2, 3, 4, 5, 6, requires_grad=True),
        torch.randn(2, 3, 16, 16, 1, 2, requires_grad=True),
        torch.randn(2, 3, 16, 16, 1, 2, 3, requires_grad=True),
        torch.randn(2, 3, 16, 16, 1, 2, 3, 4, requires_grad=True),
    ],
)
@pytest.mark.parametrize("bounds", [[-1.0, 1.0], [-0.5, 0.5], [-0.99, 0.89]])
def test_hardtanh(input_data, bounds):
    params = {"min_val": bounds[0], "max_val": bounds[1]}
    test = testing.OpTest(func=torch.nn.Hardtanh, input_args=params)
    test.check_result({"input": input_data}, train=True)
    test.check_grad_fn()


def generate_nan_tensor(shape):
    assert isinstance(shape, (list, tuple))
    t = torch.randn(shape)
    if t.dim() == 4 and random.random() < 0.5:
        t = t.to(memory_format=torch.channels_last)
    mask = torch.rand(shape) < 0.5
    t[mask] = float("nan")

    return t


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        generate_nan_tensor((10,)),
        generate_nan_tensor((10, 2)),
        generate_nan_tensor((10, 2, 2)),
        generate_nan_tensor((10, 2, 3)),
        generate_nan_tensor((10, 9, 8, 1)),
        generate_nan_tensor((10, 9, 8, 7, 2)),
        generate_nan_tensor((10, 9, 2, 2, 1, 4)),
        generate_nan_tensor((10, 0, 2, 0, 1, 4)),
        generate_nan_tensor((10, 9, 2, 2, 1, 4, 2)),
        generate_nan_tensor((10, 9, 2, 2, 1, 4, 2, 1)),
    ],
)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_isnan(input_data, dtype):
    test = testing.OpTest(func=torch.isnan, input_args={"input": input_data.to(dtype)})
    test.check_result()
    test.check_grad_fn()


# test SoftPlus
def gen_softplus_case(
    shape: Tuple[int],
    beta: int = 1,
    threshold: int = 20,
    dtype: torch.dtype = torch.float32,
):
    """
    Generate cases for softplus tests.
    """
    scale = threshold * beta
    return torch.randn(size=shape, dtype=dtype) * 2 * scale


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("beta", [1, 2, 3, 4])
@pytest.mark.parametrize("threshold", [10, 20, 30])
@pytest.mark.parametrize("test_out", [True, False])
@pytest.mark.parametrize("shape", [s.shape for s in testing.get_raw_data()])
def test_softplus(shape, dtype, beta, threshold, test_out):
    """
    SoftPlus tests.
    """
    input_args = {
        "input": gen_softplus_case(shape, beta, threshold, dtype),
        "beta": beta,
        "threshold": threshold,
    }
    if test_out:
        out = torch.tensor(np.array([]), dtype=dtype)
        input_args["out"] = out
    test = testing.OpTest(func=torch.nn.functional.softplus, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
    else:
        test.check_result()


# test softplus end


@pytest.mark.parametrize("value", testing.get_raw_data())
def test_softplus_backward(value):
    """
    SoftPlus_backward tests.
    """
    cpu_input = value
    m_input = cpu_input.clone().detach().to("musa")
    cpu_input.requires_grad = True
    m_input.requires_grad = True
    func = torch.nn.Softplus()
    func(cpu_input).sum().backward()
    func(m_input).sum().backward()
    testing.DefaultComparator()(cpu_input.grad, m_input.grad)


# test SoftPlus_backward end


@pytest.mark.parametrize("value", testing.get_raw_data())
def test_leaky_relu_backward(value):
    """
    LeakyReLu_backward tests.
    """
    cpu_input = value
    m_input = cpu_input.clone().detach().to("musa")
    cpu_input.requires_grad = True
    m_input.requires_grad = True
    func = torch.nn.LeakyReLU(0.1)
    func(cpu_input).sum().backward()
    func(m_input).sum().backward()
    testing.DefaultComparator()(cpu_input.grad, m_input.grad)


# test test_leaky_relu_backward end


# ============================== Test complex abs kernel func begin ============================== #


input_complex_datas = []
for data in testing.get_raw_data():
    if len(data.size()) < 8:
        cpu_complex = torch.randn_like(data, dtype=torch.complex64)
        input_complex_datas.append({"input": cpu_complex})


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_complex_datas)
def test_complex_abs_kernel_func(input_data):
    test = testing.OpTest(func=torch.abs, input_args=input_data)
    test.check_result()


# ============================== Test complex abs kernel func end ============================== #


unary_compare_datas = [
    # uint @ int
    {
        "input": torch.tensor([-2, -1, 0, 1, 2], dtype=torch.uint8),
        "other": 1,
    },
    # int @ int
    {
        "input": torch.tensor([-2, -1, 0, 1, 2], dtype=torch.int8),
        "other": 0,
    },
    # int @ float
    {
        "input": torch.tensor([-2, -1, 0, 1, 2], dtype=torch.int8),
        "other": 1.1,
    },
    # uint @ float
    {
        "input": torch.tensor([-2, -1, 0, 1, 2], dtype=torch.uint8),
        "other": 1.1,
    },
    # float @ int
    {
        "input": torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float16),
        "other": -1,
    },
    # float @ float
    {
        "input": torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float16),
        "other": -0.9,
    },
]
unary_compare_ops = [
    torch.eq,
    torch.ne,
    torch.gt,
    torch.ge,
    torch.lt,
    torch.le,
]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", unary_compare_datas)
@pytest.mark.parametrize("func", unary_compare_ops)
def test_unary_compare_mixed_dtypes_ops(input_data, func):
    comparator = testing.BooleanComparator()
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=comparator,
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()
