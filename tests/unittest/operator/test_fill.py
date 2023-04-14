"""Test fill operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

# Note: muDNN doesn't support float64 or bool for this operator.
# We should enable these two types after fill is implemented with MUSA.
data_type = [torch.float32, torch.int32, torch.int64]

input_data = [
    {"input": torch.rand(5, 3, 2), "value": 10},
    {"input": torch.rand(5, 3, 1, 2, 3), "value": 10},
]


@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_fill(input_data, data_type):
    test = testing.OpTest(
        func=torch.fill,
        input_args={
            "input": input_data["input"].to(data_type),
            "value": input_data["value"],
        },
    )
    test(None)
