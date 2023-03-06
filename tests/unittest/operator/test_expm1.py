"""Test expm1 operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(2,3)
    },
    {
        "input": torch.randn(3,2)
    },
    {
        "input": torch.randn(4,3)
    },
    {
        "input": torch.randn(5,4)
    },
    {
        "input": torch.randn(2,5,6)
    },
    {
        "input": torch.randn(6,5,2,3)
    },
    {
        "input": torch.randn(8,5,8,4,3)
    },
    {
        "input": torch.randn(4,5,8,4,3,2),
    },
    {
        "input": torch.randn(9,4,5,8,4,3,2),
    },
    {
        "input": torch.randn(0,9,4,5,8,4,2),
    },
    {
        "input": torch.randn(0,0,5,8,4,2),
    },
    {
        "input": torch.zeros(4,1,5,8,4,2),
    },
    {
        "input": torch.zeros(4,6,8,4,2),
    },
]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_expm1(input_data):
    test = testing.OpTest(
            func=torch.expm1,
            input_args=input_data,
            comparators=testing.DefaultComparator(abs_diff=1e-6))
    test.check_result()
