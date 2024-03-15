"""Test mv operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data_mv = [
        {
        "input": torch.randn(4, 9),
        "vec": torch.randn(9),
    },
    {
        "input": torch.randn(100, 30),
        "vec": torch.randn(30),
    },
    {
        "input": torch.randn(2, 256),
        "vec": torch.randn(256),
    },
]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data_mv", input_data_mv)
def test_mv(input_data_mv):
    test = testing.OpTest(
            func=torch.mv,
            input_args=input_data_mv,
            comparators=testing.DefaultComparator(abs_diff=1e-5))
    test.check_result()
