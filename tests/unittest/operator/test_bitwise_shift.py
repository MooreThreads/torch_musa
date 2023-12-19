"""Test bitwise shift operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
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
        "input": torch.randint(-(2**15), 2**15 - 1, (7, 9), dtype=torch.int16),
        "other": torch.randint(0, 16, (7, 9), dtype=torch.int8),
    },
    {
        "input": torch.randint(0, 255, (4, 10, 5), dtype=torch.uint8),
        "other": torch.randint(0, 8, (4, 10, 5), dtype=torch.int8),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_bitwise_right_shift(input_data):
    test = testing.OpTest(
        func=torch.bitwise_right_shift,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_bitwise_left_shift(input_data):
    test = testing.OpTest(
        func=torch.bitwise_left_shift,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
