"""Test erf and erfinv operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    {
        "input": torch.rand(0),
    },
    {
        "input": torch.rand(5),
    },
    {
        "input": torch.rand(4, 0),
    },
    {
        "input": torch.rand(10, 10),
    },
    {
        "input": torch.rand(2, 256),
    },
    {
        "input": torch.rand(16, 32, 8),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_erf(input_data):
    test = testing.OpTest(
        func=torch.special.erf,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_erf_out(input_data):
    input_tensor = input_data["input"].clone()
    data = {"input": input_tensor, "out": torch.zeros_like(input_tensor)}
    test = testing.OpTest(
        func=torch.special.erf,
        input_args=data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_erfinv(input_data):
    test = testing.OpTest(
        func=torch.special.erfinv,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_erfinv_out(input_data):
    input_tensor = input_data["input"].clone()
    data = {"input": input_tensor, "out": torch.zeros_like(input_tensor)}
    test = testing.OpTest(
        func=torch.special.erfinv,
        input_args=data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
