"""Test filp operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

data_type = testing.get_all_support_types()

inputdata = [
    {"input": torch.tensor(5.0), "dims": [0]},
    {"input": torch.randn(1), "dims": [0]},
    {"input": torch.randn(1, 2), "dims": [1]},
    {"input": torch.randn(1, 2, 3), "dims": (0, 1)},
    {"input": torch.randn(1, 2, 3), "dims": (0, 1, -1)},
    {"input": torch.randn(1, 0, 3), "dims": [0]},
    {"input": torch.randn(1, 2, 3, 4), "dims": [0, 2]},
    {"input": torch.randn(1, 2, 3, 4, 3), "dims": (0, -1)},
    {"input": torch.randn(1, 2, 3, 4, 3, 2), "dims": [2, 3]},
    {"input": torch.randn(1, 2, 3, 4, 3, 2, 4), "dims": [1, 2, 3, 4, 5]},
    {"input": torch.randn(1, 2, 3, 4, 3, 2, 4, 2), "dims": (0, 1, 2, 3, 5, 6)},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inputdata)
@pytest.mark.parametrize("data_type", data_type)
def test_flip(input_data, data_type):
    test = testing.OpTest(
        func=torch.flip,
        input_args={"input":input_data["input"].to(data_type), "dims":input_data["dims"]}
    )
    test.check_result()
