"""Test complex unary operators."""

# pylint: disable=C0116, W0611, redefined-outer-name

import copy

import pytest
import torch
import torch.nn.functional as F
import torch_musa

from torch_musa import testing


def make_complex_input(*shape):
    real = torch.rand(shape, dtype=torch.float32) * 2 - 1
    imag = torch.rand(shape, dtype=torch.float32) * 2 - 1
    return torch.complex(real, imag)


def make_complex_input_like(input_tensor):
    real = torch.rand_like(input_tensor.real, dtype=torch.float32) * 2 - 1
    imag = torch.rand_like(input_tensor.real, dtype=torch.float32) * 2 - 1
    return torch.complex(real, imag)


input_datas = [
    {"input": make_complex_input()},
    {"input": make_complex_input(1)},
    {"input": make_complex_input(128)},
    {"input": make_complex_input(16, 128)},
    {"input": make_complex_input(2, 16, 128)},
    {"input": make_complex_input(2, 4, 16, 128)},
]

all_basic_funcs = [
    torch.cos,
    torch.acos,
    torch.acosh,
    torch.asin,
    torch.asinh,
    torch.tan,
    torch.tanh,
    torch.atan,
    torch.atanh,
    torch.cosh,
    torch.exp,
    torch.exp2,
    torch.expm1,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.neg,
    torch.rsqrt,
    torch.sgn,
    torch.sigmoid,
    torch.sin,
    torch.sinc,
    torch.sinh,
    torch.sqrt,
    torch.reciprocal,
]

all_inplace_funcs = [
    "cos_",
    "acos_",
    "acosh_",
    "asin_",
    "asinh_",
    "tan_",
    "tanh_",
    "atan_",
    "atanh_",
    "cosh_",
    "exp_",
    "exp2_",
    "expm1_",
    "log_",
    "log10_",
    "log1p_",
    "log2_",
    "neg_",
    "rsqrt_",
    "sqrt_",
    "reciprocal_",
]

all_activation_funcs = [
    F.silu,
    F.tanhshrink,
]

all_activation_inplace_funcs = [
    F.silu,
]

ACT_TEST_TOLERANCE = {
    # torch.complex32: (5e-2, 5e-3),
    torch.complex64: (1e-4, 1e-4),
    torch.complex128: (1e-6, 1e-6),
}

complex_dtypes = [torch.complex64, torch.complex128]


def function(input_data, dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data_new = copy.deepcopy(input_data)
        input_data_new["input"] = make_complex_input_like(input_data_new["input"]).to(
            dtype
        )
    if "out" in input_data.keys() and isinstance(input_data["out"], torch.Tensor):
        input_data_new["out"] = input_data_new["out"].to(dtype)

    abs_diff, rel_diff = ACT_TEST_TOLERANCE.get(
        dtype, ACT_TEST_TOLERANCE[torch.complex64]
    )
    comparator = testing.DefaultComparator(abs_diff, rel_diff, equal_nan=True)
    test = testing.OpTest(
        func=func,
        input_args=input_data_new,
        comparators=comparator,
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", complex_dtypes)
@pytest.mark.parametrize("func", all_basic_funcs)
def test_all_basic_funcs(input_data, dtype, func):
    function(input_data, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", complex_dtypes)
@pytest.mark.parametrize("func_name", all_inplace_funcs)
def test_all_inplace_funcs(input_data, dtype, func_name):
    self_tensor = make_complex_input_like(input_data["input"]).to(dtype)
    abs_diff, rel_diff = ACT_TEST_TOLERANCE.get(
        dtype, ACT_TEST_TOLERANCE[torch.complex64]
    )
    test = testing.InplaceOpChek(
        func_name=func_name,
        self_tensor=self_tensor,
        comparators=[testing.DefaultComparator(abs_diff, rel_diff, equal_nan=True)],
    )
    test.check_res()
    test.check_address()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", complex_dtypes)
@pytest.mark.parametrize("func", all_activation_inplace_funcs)
def test_all_activation_inplace_funcs(input_data, dtype, func):
    self_tensor = make_complex_input_like(input_data["input"]).to(dtype)
    musa_self_tensor = self_tensor.clone().musa()
    cpu_self_tensor = self_tensor.clone().cpu()
    musa_addr = musa_self_tensor.data_ptr()
    cpu_addr = cpu_self_tensor.data_ptr()

    musa_res = func(musa_self_tensor, inplace=True)
    cpu_res = func(cpu_self_tensor, inplace=True)

    assert musa_self_tensor.data_ptr() == musa_addr
    assert cpu_self_tensor.data_ptr() == cpu_addr
    assert musa_res.data_ptr() == musa_addr
    assert cpu_res.data_ptr() == cpu_addr

    abs_diff, rel_diff = ACT_TEST_TOLERANCE.get(
        dtype, ACT_TEST_TOLERANCE[torch.complex64]
    )
    comparator = testing.DefaultComparator(abs_diff, rel_diff, equal_nan=True)
    assert comparator(musa_self_tensor.cpu(), cpu_self_tensor)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", complex_dtypes)
@pytest.mark.parametrize("func", all_activation_funcs)
def test_all_activation_funcs(input_data, dtype, func):
    function(input_data, dtype, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", complex_dtypes)
@pytest.mark.parametrize("func", all_basic_funcs)
def test_all_basic_funcs_out(input_data, dtype, func):
    input_data = copy.deepcopy(input_data)
    input_data["out"] = torch.tensor([])
    function(input_data, dtype, func)
