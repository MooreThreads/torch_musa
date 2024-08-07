"""Test mv operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data_mv = [
    {
        "input": torch.randn(4, 0),
        "vec": torch.randn(0),
    },
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
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data_mv", input_data_mv)
def test_mv_fp16(input_data_mv):
    test = testing.OpTest(
        func=torch.mv,
        input_args=input_data_mv,
        comparators=testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3),
    )
    test.check_musafp16_vs_musafp32()
    test.check_out_ops(fp16=True)
    test.check_grad_fn(fp16=True)
