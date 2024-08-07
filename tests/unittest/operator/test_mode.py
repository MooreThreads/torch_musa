"""Test mode operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
from torch_musa import testing


def function(input_data, dim, keepdim, func):
    input_data = {
        "input": input_data,
        "dim": dim,
        "keepdim": keepdim,
    }
    test = testing.OpTest(func=func, input_args=input_data)
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.rand(3, 4), "dim": -1},
        {"input": torch.rand(10, 2), "dim": 1},
        {"input": torch.rand(3, 4, 6), "dim": 0},
        {"input": torch.rand(3, 4, 6), "dim": 1},
        {"input": torch.rand(8, 10, 1, 20), "dim": 2},
        {"input": torch.rand(8, 10, 16, 20), "dim": 3},
        {"input": torch.rand(4, 8, 10, 16, 20), "dim": 1},
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_mode(input_data, keepdim):
    function(input_data["input"], input_data["dim"], keepdim, torch.mode)
