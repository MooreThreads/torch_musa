"""Test roll operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

data_type = testing.get_all_support_types()

inputdata = [
    {"input": torch.randn(4, 2), "shifts": (2, 1), "dims": (0, 1)},
    {"input": torch.randn(1, 2, 3), "shifts": 1, "dims": 1},
    {"input": torch.randn(4, 5, 6, 7), "shifts": (2, 1), "dims": (0, 1)},
    {"input": torch.randn(1, 2, 3, 4, 3), "shifts": (3, 2, 1), "dims": (0, 1, 2)},
    {"input": torch.randn(1, 2, 3, 4, 3, 2), "shifts": 1, "dims": 1},
    {"input": torch.randn(1, 2, 3, 4, 3, 2, 4), "shifts": (4, 3, 2, 1), "dims": (0, 1, 2, 3)},
    {"input": torch.randn(1, 2, 3, 4, 3, 2, 4, 2), "shifts": 1, "dims": 1},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inputdata)
@pytest.mark.parametrize("data_type", data_type)
def test_roll(input_data, data_type):
    test = testing.OpTest(
        func=torch.roll,
        input_args={"input":input_data["input"].to(data_type),
                    "shifts": input_data["shifts"],
                    "dims":input_data["dims"]}
    )
    test.check_result()
