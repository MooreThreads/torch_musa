"""Test bitwise shift operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, redefined-builtin
import copy
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randint(-128, 127, (4, 10), dtype=torch.int8),
        "other": 4,
    },
    {
        "input": torch.randint(-(2**31), 2**31 - 1, (12, 6), dtype=torch.int32),
        "other": 8,
    },
    {
        "input": torch.randint(-128, 127, (4, 10, 5), dtype=torch.int8),
        "other": torch.randint(0, 8, (4, 10, 5), dtype=torch.int8),
    },
    {
        "input": torch.randint(-128, 127, (4, 4), dtype=torch.int8),
        "other": torch.randint(0, 8, (4, 4), dtype=torch.int8),
    },
    {
        "input": torch.randint(-(2**31), 2**31 - 1, (12, 6), dtype=torch.int32),
        "other": torch.randint(0, 32, (12, 6), dtype=torch.int8),
    },
    {
        "input": torch.randint(-(2**31), 2**31 - 1, (12, 0), dtype=torch.int32),
        "other": torch.randint(0, 32, (12, 0), dtype=torch.int8),
    },
    {
        "input": torch.randint(-(2**15), 2**15 - 1, (7, 9), dtype=torch.int16),
        "other": torch.randint(0, 16, (7, 9), dtype=torch.int8),
    },
    {
        "input": torch.randint(0, 255, (4, 10, 5), dtype=torch.uint8),
        "other": torch.randint(0, 8, (4, 10, 5), dtype=torch.int8),
    },
    {
        "input": torch.randint(0, 255, (0, 10, 5), dtype=torch.uint8),
        "other": torch.randint(0, 8, (0, 10, 5), dtype=torch.int8),
    },
]


def rshift_wrapper(input, other, out=None):
    result = input >> other
    if out is not None:
        out.copy_(result)
        return out
    return result


def lshift_wrapper(input, other, out=None):
    result = input << other
    if out is not None:
        out.copy_(result)
        return out
    return result


def irshift_wrapper(input, other):
    input >>= other
    return input


def ilshift_wrapper(input, other):
    input <<= other
    return input


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("func", [torch.bitwise_right_shift, rshift_wrapper])
def test_bitwise_right_shift(input_data, func):
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_bitwise_right_shift_inplace(input_data):
    inplace_input = copy.deepcopy(input_data)
    test = testing.InplaceOpChek(
        func_name=torch.bitwise_right_shift.__name__ + "_",
        self_tensor=inplace_input["input"],
        input_args={"other": inplace_input["other"]},
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("func", [torch.bitwise_left_shift, lshift_wrapper])
def test_bitwise_left_shift(input_data, func):
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_bitwise_left_shift_inplace(input_data):
    inplace_input = copy.deepcopy(input_data)
    test = testing.InplaceOpChek(
        func_name=torch.bitwise_left_shift.__name__ + "_",
        self_tensor=inplace_input["input"],
        input_args={"other": inplace_input["other"]},
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("func", [irshift_wrapper])
def test_bitwise_right_shift_inplace2(input_data, func):
    inplace_input = copy.deepcopy(input_data)
    test = testing.OpTest(
        func=func,
        input_args=inplace_input,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("func", [ilshift_wrapper])
def test_bitwise_left_shift_inplace2(input_data, func):
    inplace_input = copy.deepcopy(input_data)
    test = testing.OpTest(
        func=func,
        input_args=inplace_input,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()


@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("func", [irshift_wrapper, ilshift_wrapper])
def test_bitwise_shift_inplace_device_guard(input_data, func):
    inplace_input1 = copy.deepcopy(input_data)
    inplace_input1["input"].to("musa")
    func(**inplace_input1)
    inplace_input2 = copy.deepcopy(input_data)
    inplace_input2["input"].to("musa")
    current_device = torch.musa.current_device()
    torch.musa.set_device((current_device + 1) % torch.musa.device_count())
    func(**inplace_input2)
    assert inplace_input1["input"].device == inplace_input2["input"].device
    assert torch.allclose(
        inplace_input1["input"], inplace_input2["input"], equal_nan=True
    )
    torch.musa.set_device(current_device)
