"""Test bmm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(4, 10, 5),
        "mat2": torch.randn(4, 5, 10),
    },
    {
        "input": torch.randn(4, 10, 0),
        "mat2": torch.randn(4, 0, 10),
    },
    {
        "input": torch.randn(4, 4).T.unsqueeze(0),
        "mat2": torch.randn(4, 4).T.unsqueeze(0),
    },
    {
        "input": torch.randn(6, 4).T.unsqueeze(0),
        "mat2": torch.randn(4, 6).T.unsqueeze(0),
    },
    {
        "input": torch.randn(7, 9).T.unsqueeze(0),
        "mat2": torch.randn(9, 7).T.unsqueeze(0),
    },
    {
        "input": torch.randn(4, 10, 5).transpose(1, 2),
        "mat2": torch.randn(4, 5, 10).transpose(1, 2),
    },
    {
        "input": torch.randn(4, 10, 5).transpose(0, 1),
        "mat2": torch.randn(5, 10, 10).transpose(0, 1),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_bmm(input_data):
    test = testing.OpTest(
        func=torch.bmm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


input_data_complex = [
    {
        "input": torch.randn(4, 10, 5, dtype=torch.complex64),
        "mat2": torch.randn(4, 5, 10, dtype=torch.complex64),
    },
    {
        "input": torch.randn(4, 10, 0, dtype=torch.complex128),
        "mat2": torch.randn(4, 0, 10, dtype=torch.complex128),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data_complex)
def test_bmm_complex(input_data):
    test = testing.OpTest(
        func=torch.bmm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()
