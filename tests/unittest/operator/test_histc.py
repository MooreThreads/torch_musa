"""Test prod operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
from torch_musa import testing


def function(input_data, bins, vmin, vmax, func):
    input_data = {
        "input": input_data["input"],
        "bins": bins,
        "min": vmin,
        "max": vmax,
    }
    test = testing.OpTest(func=func, input_args=input_data)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randint(0, 20, (3,)).to(torch.float)},
        {"input": torch.randint(0, 20, (10,)).to(torch.float)},
        {"input": torch.randint(0, 20, (20,)).to(torch.float)},
        {"input": torch.randint(0, 20, (20, 5)).to(torch.float)},
    ],
)
@pytest.mark.parametrize("bins", [5, 10, 20])
def test_histc(input_data, bins):
    vmin, vmax = 0, 20
    function(input_data, bins, vmin, vmax, torch.histc)
