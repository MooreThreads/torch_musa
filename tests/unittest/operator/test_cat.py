"""Test cat operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

inputdata = [
    {
        "input": [torch.randn(1), torch.randn(1)],
        "dim": 0,
    },
    {
        "input": [torch.randn(1, 2), torch.randn(1, 2)],
        "dim": 1,
    },
    {
        "input": [torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
        ],
        "dim": 3,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inputdata)
def test_cat(input_data):
    inputs = {"tensors": input_data["input"], "dim": input_data["dim"]}
    test = testing.OpTest(
        func=torch.cat,
        input_args=inputs,
    )
    test.check_result()
