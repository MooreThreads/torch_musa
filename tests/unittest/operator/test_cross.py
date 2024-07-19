"""Test cross operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(0, 3),
        "other": torch.randn(0, 3),
    },
    {
        "input": torch.randn(0, 0, 3),
        "other": torch.randn(0, 0, 3),
    },
    {
        "input": torch.randn(0, 0, 0, 3),
        "other": torch.randn(0, 0, 0, 3),
    },
    {
        "input": torch.randn(4, 3),
        "other": torch.randn(4, 3),
    },
    {"input": torch.randn(1024, 23, 3), "other": torch.randn(1024, 23, 3), "dim": 2},
    {"input": torch.randn(2, 3), "other": torch.randn(2, 3), "dim": 1},
    {
        "input": torch.randn(2, 2, 2, 3),
        "other": torch.randn(2, 2, 2, 3),
    },
    {
        "input": torch.randn(2, 1, 2, 3).to(memory_format=torch.channels_last),
        "other": torch.randn(2, 1, 2, 3).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(2, 3, 1, 1).to(memory_format=torch.channels_last),
        "other": torch.randn(2, 3, 1, 1).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last),
        "other": torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(3),
        "other": torch.randn(3),
    },
    {
        "input": torch.randn(2, 3, 2, 6, 3),
        "other": torch.randn(2, 3, 2, 6, 3),
    },
    {
        "input": torch.randn(2, 3, 2, 6, 9, 3),
        "other": torch.randn(2, 3, 2, 6, 9, 3),
    },
    {
        "input": torch.randn(2, 3, 2, 6, 9, 3, 5),
        "other": torch.randn(2, 3, 2, 6, 9, 3, 5),
    },
    {
        "input": torch.randn(2, 3, 2, 6, 9, 7, 3, 5),
        "other": torch.randn(2, 3, 2, 6, 9, 7, 3, 5),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_cross(input_data):
    test = testing.OpTest(
        func=torch.cross,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()
