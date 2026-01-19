"""Test torch.special.* operators on MUSA """

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    {
        "input": torch.rand(0),
    },
    {
        "input": torch.rand(5),
    },
    {
        "input": torch.rand(4, 0),
    },
    {
        "input": torch.rand(10, 10),
    },
    {
        "input": torch.rand(2, 256),
    },
    {
        "input": torch.rand(16, 32, 8),
    },
]

float_dtypes = [torch.float32, torch.float16]
# bf16 is not supported on arch older than qy2
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_i0(input_data):
    test = testing.OpTest(
        func=torch.i0,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_i0_(input_data):
    test = testing.InplaceOpChek(
        func_name=torch.i0.__name__ + "_",
        self_tensor=input_data["input"],
        comparators=[testing.DefaultComparator(abs_diff=1e-3)],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("float_dtype", float_dtypes)
def test_i0_out(input_data, float_dtype):
    input_tensor = input_data["input"].clone().to(float_dtype)
    data = {"input": input_tensor, "out": torch.zeros_like(input_tensor)}
    test = testing.OpTest(
        func=torch.i0,
        input_args=data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_airy_ai_out(input_data):
    input_tensor = input_data["input"].clone().to(torch.float32)
    data = {"x": input_tensor, "out": torch.zeros_like(input_tensor)}
    test = testing.OpTest(
        func=torch.special.airy_ai,
        input_args=data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()


################## test_special_unary ##################
special_op = [
    torch.special.entr,
    torch.special.ndtri,
    torch.special.log_ndtr,
    torch.special.erfcx,
    torch.special.i0e,
    torch.special.i1,
    torch.special.i1e,
    torch.special.bessel_j0,
    torch.special.bessel_j1,
    torch.special.bessel_y0,
]

# TODO: qy2 bessel_y1 precision error
if testing.get_musa_arch() > 22:
    special_op.append(torch.special.bessel_y1)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("func_op", special_op)
def test_special_unary(input_data, func_op):
    test = testing.OpTest(
        func=func_op,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("func_op", special_op)
def test_special_out_unary(input_data, func_op):
    input_tensor = input_data["input"].clone()
    data = {"input": input_tensor, "out": torch.zeros_like(input_tensor)}
    test = testing.OpTest(
        func=func_op,
        input_args=data,
        comparators=testing.DefaultComparator(abs_diff=1e-2),
    )
    test.check_result()


################## test_special_binary ##################
input_datas = [
    {"input": torch.randint(2, 10, (4,)), "other": torch.randint(2, 10, (4,))},
    {"input": torch.randint(2, 10, (4, 5)), "other": torch.randint(2, 10, (4, 5))},
    {
        "input": torch.randint(2, 10, (4, 5, 6)),
        "other": torch.randint(2, 10, (4, 5, 6)),
    },
]

special_op = [torch.special.xlog1py, torch.special.zeta]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("func_op", special_op)
def test_special_binary(input_data, func_op):
    test = testing.OpTest(
        func=func_op,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("func_op", special_op)
def test_special_out_binary(input_data, func_op):
    input_tensor = input_data["input"].clone().float()
    other_tensor = input_data["other"].clone().float()
    data = {
        "input": input_tensor,
        "other": other_tensor,
        "out": torch.zeros_like(input_tensor),
    }
    test = testing.OpTest(
        func=func_op,
        input_args=data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
