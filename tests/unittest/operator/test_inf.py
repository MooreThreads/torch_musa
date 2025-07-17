"""Test inf operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        (torch.tensor([-float("inf"), float("inf"), 1.2])),
        (torch.tensor([0, float("-inf"), float("inf")])),
        (torch.tensor([[-float("inf"), float("inf")], [-float("inf"), 1.0]])),
    ],
)
def test_isposinf(input_data):
    test = testing.OpTest(
        func=torch.isposinf,
        input_args={"input": input_data},
    )
    test.check_result()
    test.check_out_ops()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        (torch.tensor([-float("inf"), float("inf"), 1.2])),
        (torch.tensor([0, float("-inf"), float("inf")])),
        (torch.tensor([[-float("inf"), float("inf")], [-float("inf"), 1.0]])),
    ],
)
def test_isneginf(input_data):
    test = testing.OpTest(
        func=torch.isneginf,
        input_args={"input": input_data},
    )
    test.check_result()
    test.check_out_ops()
