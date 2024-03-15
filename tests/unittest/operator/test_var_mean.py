"""Test var_mean operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

input_data = [
    {"input": torch.randn([1, 10]), "correction": 0, "keepdim": True, "dim": 1},
    {"input": torch.randn([1, 10, 5]), "correction": 0, "keepdim": False, "dim": 2},
    {"input": torch.randn([1, 10, 5, 5]), "correction": 1, "keepdim": True, "dim": 3},
    {"input": torch.randn([1, 10, 5, 5, 10]), "correction": 1, "keepdim": False, "dim": 4},
    {"input": torch.randn([9, 8, 7, 6, 5, 4]), "correction": 0, "keepdim": True, "dim": 5},
    {"input": torch.randn([9, 8, 7, 6, 5, 4, 16]), "correction": 0, "keepdim": False, "dim": 5},
    {"input": torch.randn([9, 8, 7, 6, 5, 4, 5, 20]), "correction": 1, "keepdim": True, "dim": 7}
]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", [torch.float32])
def test_var_mean(input_data, data_type):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(data_type)
    test = testing.OpTest(
        func=torch.var_mean,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-5)
    )
    test.check_result()
