"""Test conj operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
from torch_musa import testing

input_data = [
    {
        "input": torch.randn(5, 3, dtype=torch.bfloat16)
        + 1j * torch.randn(5, 3, dtype=torch.bfloat16)
    },
    {
        "input": torch.randn(5, 3, dtype=torch.float32)
        + 1j * torch.randn(5, 3, dtype=torch.float32)
    },
    {
        "input": torch.randn(10, 5, 3, dtype=torch.float16)
        + 1j * torch.randn(10, 5, 3, dtype=torch.float16)
    },
    {
        "input": torch.randn(10, 5, 3, dtype=torch.float32)
        + 1j * torch.randn(10, 5, 3, dtype=torch.float32)
    },
    {
        "input": torch.randn(4096, 2048, dtype=torch.float32)
        + 1j * torch.randn(4096, 2048, dtype=torch.float32)
    },
    {
        "input": torch.randn(1, dtype=torch.float32)
        + 1j * torch.randn(1, dtype=torch.float32)
    },
    {
        "input": torch.randn(1, dtype=torch.float64)
        + 1j * torch.randn(1, dtype=torch.float64)
    },
    {
        "input": torch.randn(4096, 2048, dtype=torch.float64)
        + 1j * torch.randn(4096, 2048, dtype=torch.float64)
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_conj(input_data):
    # gen cpu res
    cpu_input = input_data["input"]
    cpu_out = torch.conj(cpu_input)

    # gen musa res
    musa_input = input_data["input"].musa()
    musa_out = torch.conj(musa_input)

    test = testing.OpTest(
        func=torch.conj,
        input_args={"input": input_data["input"]},
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.compare_res(cpu_out, musa_out.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_conj_physical(input_data):
    # gen cpu res
    cpu_input = input_data["input"]
    cpu_out = torch.conj_physical(cpu_input)

    # gen musa res
    musa_input = input_data["input"].musa()
    musa_out = torch.conj_physical(musa_input)

    test = testing.OpTest(
        func=torch.conj_physical,
        input_args={"input": input_data["input"]},
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.compare_res(cpu_out, musa_out.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_is_conj(input_data):
    # gen cpu res
    cpu_input = input_data["input"]
    cpu_conj = torch.conj(cpu_input)
    cpu_is_conj = torch.is_conj(cpu_conj)

    # gen musa res
    musa_input = input_data["input"].musa()
    musa_conj = torch.conj(musa_input)
    musa_is_conj = torch.is_conj(musa_conj)
    assert cpu_is_conj == musa_is_conj
