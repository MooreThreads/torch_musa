"""Test fill operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

# Note: muDNN doesn't support float64 or bool for this operator.
# We should enable these two types after fill is implemented with MUSA.
data_type = testing.get_all_support_types()
input_data = [
    {"input": torch.rand(5, 3, 2), "value": 10},
    {"input": torch.rand(5, 3, 1, 2, 3), "value": 10},
]

for data in testing.get_raw_data():
    input_data.append({"input": data, "value": 10})


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
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
    test.check_result()
