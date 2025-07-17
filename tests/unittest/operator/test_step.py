"""Test step operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data,other_data",
    [
        (torch.randn(2, 2), torch.randn(2, 2)),
        (torch.randn(3, 3, 3), torch.randn(3, 3, 3)),
        (torch.randn(4, 4, 4, 4), torch.randn(4, 4, 4, 4)),
    ],
)
def test_nextafter(input_data, other_data):
    test = testing.OpTest(
        func=torch.nextafter,
        input_args={"input": input_data, "other": other_data},
    )
    test.check_result()
    test.check_out_ops()
