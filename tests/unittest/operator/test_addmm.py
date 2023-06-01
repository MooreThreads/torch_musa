"""Test addmm operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(4, 2),
        "mat1": torch.randn(4, 0),
        "mat2": torch.randn(0, 2),
        "beta": 1,
        "alpha": 1.2,
    },
    {
        "input": torch.randn(0, 2),
        "mat1": torch.randn(0, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(2, 2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1,
        "alpha": 1,
    },
    {
        "input": torch.randn(2, 2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.7,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(2, 2),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.tensor(112.5),
        "mat1": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(5, 1),
        "mat1": torch.randn(30, 5).t(),
        "mat2": torch.randn(30, 5),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(1, 5),
        "mat1": torch.randn(5, 30),
        "mat2": torch.randn(30, 5),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(1, 5),
        "mat1": torch.randn(30, 5).t(),
        "mat2": torch.randn(5, 30).t(),
        "beta": 1.2,
        "alpha": 1.7,
    },
    {
        "input": torch.randn(1, 5),
        "mat1": torch.randn(5, 30),
        "mat2": torch.randn(5, 30).t(),
        "beta": 1.2,
        "alpha": 1.7,
    },
]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_addmm(input_data):
    test = testing.OpTest(
            func=torch.addmm,
            input_args=input_data,
            comparators=testing.DefaultComparator(abs_diff=1e-5))
    test.check_result()
