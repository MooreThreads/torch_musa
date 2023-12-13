"""Test dot operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(4),
        "tensor": torch.randn(4),
    },
    {
        "input": torch.randn(1024),
        "tensor": torch.randn(1024),
    },
    {
        "input": torch.randn(256),
        "tensor": torch.randn(256),
    },
    {
        "input": torch.randn(256)[1:20:2],
        "tensor": torch.randn(256)[1:20:2],
    },
]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_dot(input_data):
    test = testing.OpTest(
            func=torch.dot,
            input_args=input_data,
            comparators=testing.DefaultComparator(abs_diff=1e-6))
    test.check_result()
