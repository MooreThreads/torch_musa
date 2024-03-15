"""Test prod operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
from torch_musa import testing


def function(input_data, dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(func=func, input_args=input_data)
    test.check_result()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([1, 10]), "dim": 1},
        {"input": torch.randn([1, 10, 5]), "dim": 2},
        {"input": torch.randn([1, 10, 5, 5]), "dim": 3},
        {"input": torch.randn([1, 10, 5, 5, 10]), "dim": 4},
        {"input": torch.randn([9, 8, 7, 6, 5, 4]), "dim": 5},
        {"input": torch.randn([9, 8, 7, 6, 5, 4, 3]), "dim": 6},
        {"input": torch.randn([9, 8, 7, 6, 5, 4, 5, 20]), "dim": 7}
    ])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_prod(input_data, dtype):
    function(input_data, dtype, torch.prod)
