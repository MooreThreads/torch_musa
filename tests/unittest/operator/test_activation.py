"""Test activation operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from typing import Tuple
import numpy as np
import torch
import pytest
import torch_musa
from torch_musa import testing

input_datas = [
    {"input": torch.randn(60)},
    {"input": torch.randn(60, 1)},
]

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
    torch.logical_not
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
    torch.floor_
]


all_nn_funcs = [
    torch.nn.ReLU(),
    # torch.nn.GELU(approximate="none"),
    torch.nn.SiLU(),
    torch.nn.LeakyReLU(),
    torch.nn.Hardswish(),
    torch.nn.Hardsigmoid(),
]


def function(input_data, dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    if "out" in input_data.keys() and isinstance(input_data["out"], torch.Tensor):
        input_data["out"] = input_data["out"].to(dtype)
    if "min" in input_data.keys() and isinstance(input_data["min"], torch.Tensor):
        input_data["min"] = input_data["min"].to(dtype)
    if "max" in input_data.keys() and isinstance(input_data["max"], torch.Tensor):
        input_data["max"] = input_data["max"].to(dtype)
    if func in (torch.log, torch.log_):
        test = testing.OpTest(func=func, input_args=input_data,
                              comparators=testing.DefaultComparator(abs_diff=1e-6, equal_nan=True))
    else:
        test = testing.OpTest(func=func, input_args=input_data)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("func", all_basic_funcs + all_inplace_funcs)
def test_all_basic_funcs(input_data, dtype, func):
    function(input_data, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("func", all_basic_funcs)
def test_all_basic_funcs_out(input_data, dtype, func):
    out = torch.tensor(np.array([]))
    input_args = {"input": input_data["input"], "out": out}
    function(input_args, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
# torch.bitwise_not only support integral and Boolean type
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("func", [torch.bitwise_not])
def test_bitwise_not(input_data, dtype, func):
    function(input_data, dtype, func)


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_neg(input_data, dtype):
    function(input_data, dtype, torch.neg)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_neg_out(input_data, dtype):
    out = torch.tensor(np.array([]))
    input_args = {"input": input_data["input"], "out": out}
    function(input_args, dtype, torch.neg)


# =================================== Test torch.neg end =================================== #


# =================================== Test nn functions begin =================================== #
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("func", all_nn_funcs)
def test_nn_funcs(input_data, dtype, func):
    function(input_data, dtype, func)


# =================================== Test nn functions end ================================== #


# =================================== Test torch.nn.GELU begin =================================== #
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("func", [torch.nn.GELU(approximate="none")])
def test_gelu(input_data, dtype, func):
    input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()


# =================================== Test torch.nn.GELU end =================================== #


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
    {"input": torch.tensor(35), "min": torch.tensor(34),
     "max": torch.tensor(77)},
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
]

min_value = [-50, -40]
max_value = [40, 50]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_min", min_value)
@pytest.mark.parametrize("_max", max_value)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("func", [torch.clamp])
def test_clamp_min_max(input_data, _min, _max, dtype, func):
    input_args = {
        "input": input_data["input"],
        "min": _min,
        "max": _max,
    }
    function(input_args, dtype, func)

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
    test = testing.OpTest(func=torch.clamp, input_args=input_args,
                          comparators=testing.DefaultComparator(abs_diff=1e-5))
    test.check_musafp16_vs_musafp32()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_min", min_value)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
# @pytest.mark.parametrize("func", [torch.clamp, torch.clamp_min])
@pytest.mark.parametrize("func", [torch.clamp])
def test_clamp_min(input_data, _min, dtype, func):
    input_args = {"input": input_data["input"], "min": _min}
    function(input_args, dtype, func)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_min", min_value)
def test_clamp_min_fp16(input_data, _min):
    input_args = {
        "input": input_data["input"].to(torch.float16).to(torch.float32),
        "min": _min,
    }
    test = testing.OpTest(func=torch.clamp, input_args=input_args,
                          comparators=testing.DefaultComparator(abs_diff=1e-5))
    test.check_musafp16_vs_musafp32()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_max", max_value)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("func", [torch.clamp])
def test_clamp_max(input_data, _max, dtype, func):
    input_args = {"input": input_data["input"], "max": _max}
    function(input_args, dtype, func)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_max", max_value)
def test_clamp_max_fp16(input_data, _max):
    input_args = {
        "input": input_data["input"].to(torch.float16).to(torch.float32),
        "max": _max,
    }
    test = testing.OpTest(func=torch.clamp, input_args=input_args,
                          comparators=testing.DefaultComparator(abs_diff=1e-5))
    test.check_musafp16_vs_musafp32()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("_min", min_value)
@pytest.mark.parametrize("_max", max_value)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("func", [torch.clamp])
def test_clamp_scalar_min_max_out(input_data, _min, _max, dtype, func):
    out = torch.tensor(np.array([]))
    input_args = {"input": input_data["input"],
                  "min": _min, "max": _max, "out": out}
    function(input_args, dtype, func)


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


# =================================== Test torch.pow begin =================================== #
input_datas = [
    {"input": torch.randn(60), "exponent": 1.0},
    {"input": torch.randn(60, 2), "exponent": 2.0},
    {"input": torch.randn(60, 2, 3), "exponent": 3.0},
    {"input": torch.randn(60, 2, 3, 4), "exponent": 4.0},
    {"input": torch.arange(0, 24 * 5).reshape(1, 2, 3,
                                              4, -1), "exponent": 2.0},
    {"input": torch.arange(0, 24 * 5).reshape(1, 2, 3,
                                              4, 5, -1), "exponent": 1.0},
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
]


# torch.pow only support float32
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pow(input_data, dtype):
    function(input_data, dtype, torch.pow)


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
@pytest.mark.parametrize("input_data",
                         [
                             torch.randn(2, requires_grad=True),
                             torch.randn(2, 3, 16, 16, requires_grad=True),
                             torch.randn(2, 3, 4, 5, 6, requires_grad=True),
                             torch.randn(2, 3, 16, 16, 1, 2,
                                         requires_grad=True),
                             torch.randn(2, 3, 16, 16, 1, 2, 3,
                                         requires_grad=True),
                             torch.randn(2, 3, 16, 16, 1, 2, 3,
                                         4, requires_grad=True),
                         ]
                         )
@pytest.mark.parametrize("bounds", [[-1.0, 1.0], [-0.5, 0.5], [-0.99, 0.89]])
def test_hardtanh(input_data, bounds):
    params = {"min_val": bounds[0],
              "max_val": bounds[1]}
    test = testing.OpTest(func=torch.nn.Hardtanh, input_args=params)
    test.check_result({"input": input_data}, train=True)


def generate_nan_tensor(shape):
    assert isinstance(shape, (list, tuple))
    t = torch.randn(shape)
    mask = torch.rand(shape) < 0.5
    t[mask] = float("nan")

    return t


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data",
                         [
                             generate_nan_tensor((10, )),
                             generate_nan_tensor((10, 2)),
                             generate_nan_tensor((10, 2, 2)),
                             generate_nan_tensor((10, 2, 3)),
                             generate_nan_tensor((10, 9, 8, 1)),
                             generate_nan_tensor((10, 9, 8, 7, 2)),
                             generate_nan_tensor((10, 9, 2, 2, 1, 4)),
                             generate_nan_tensor((10, 9, 2, 2, 1, 4, 2)),
                             generate_nan_tensor((10, 9, 2, 2, 1, 4, 2, 1)),
                         ]

                         )
@pytest.mark.parametrize("dtype", [torch.float32])
def test_isnan(input_data, dtype):
    test = testing.OpTest(func=torch.isnan, input_args={
                          "input": input_data.to(dtype)})
    test.check_result()


# test SoftPlus
def gen_sfotplus_case(shape: Tuple[int],
                      beta: int = 1,
                      threshold: int = 20,
                      dtype: torch.dtype = torch.float32):
    """
     Generate cases for softplus tests.
    """
    scale = threshold * beta
    return torch.randn(size=shape, dtype=dtype) * 2 * scale


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("beta", [1, 2, 3, 4])
@pytest.mark.parametrize("threshold", [10, 20, 30])
@pytest.mark.parametrize("test_out", [True, False])
@pytest.mark.parametrize("shape", [s.shape for s in testing.get_raw_data()])
def test_softplus(shape, dtype, beta, threshold, test_out):
    """
    SoftPlus tests.
    """
    input_args = {"input": gen_sfotplus_case(
        shape, beta, threshold, dtype), "beta": beta, "threshold": threshold}
    if test_out:
        out = torch.tensor(np.array([]), dtype=dtype)
        input_args["out"] = out
    test = testing.OpTest(
        func=torch.nn.functional.softplus, input_args=input_args)
    test.check_result()
# test softplus end


@pytest.mark.parametrize("value", testing.get_raw_data())
def test_softplus_backward(value):
    """
    SoftPlus_backward tests.
    """
    cpu_input =value
    m_input = cpu_input.clone().detach().to('musa')
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
    cpu_input =value
    m_input = cpu_input.clone().detach().to('musa')
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
