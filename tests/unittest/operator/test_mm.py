"""Test addmm operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing
input_data = [
            {
             'input': torch.randn(4, 0),
             'mat2': torch.randn(0, 2),
             },
            {
             'input': torch.randn(0, 30),
             'mat2': torch.randn(30, 2),
             },
            {
             'input': torch.randn(2, 30),
             'mat2': torch.randn(30, 2),
             },
             {
             'input': torch.randn(30, 5).t(),
             'mat2': torch.randn(30, 2),
             },
             {
             'input': torch.randn(30, 5).t(),
             'mat2': torch.randn(2, 30).t(),
             },
             {
             'input': torch.randn(5, 30),
             'mat2': torch.randn(2, 30).t(),
             },
    ]



@pytest.mark.parametrize("input_data", input_data)
def test_mm(input_data):
    test = testing.OpTest(
        func=torch.mm,
        input_args=input_data
    )
    test(None)
