"""Test mm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(4, 0),
        "mat2": torch.randn(0, 2),
    },
    {
        "input": torch.randn(0, 30),
        "mat2": torch.randn(30, 2),
    },
    {
        "input": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
    },
    {
        "input": torch.randn(30, 5).t(),
        "mat2": torch.randn(30, 2),
    },
    {
        "input": torch.randn(30, 5).t(),
        "mat2": torch.randn(2, 30).t(),
    },
    {
        "input": torch.randn(5, 30),
        "mat2": torch.randn(2, 30).t(),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_mm(input_data):
    test = testing.OpTest(
        func=torch.mm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_mm_fp16(input_data):
    test = testing.OpTest(
        func=torch.mm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3),
    )
    test.check_musafp16_vs_musafp32()
    test.check_out_ops(fp16=True)
    test.check_grad_fn(fp16=True)
