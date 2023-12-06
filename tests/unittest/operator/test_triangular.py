"""Test filp operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

data_type = testing.get_all_types()
input_data = [
    {"input": torch.rand(5, 3), "diagonal": 0},
    {"input": torch.rand(5, 3), "diagonal": 1},
    {"input": torch.rand(5, 3), "diagonal": -1},
    {"input": torch.rand(5, 3, 1), "diagonal": 0},
    {"input": torch.rand(5, 3, 3), "diagonal": 1},
    {"input": torch.rand(5, 3, 3, 2), "diagonal": 1},
    {"input": torch.rand(5, 3, 3, 6, 3), "diagonal": 2},
    {"input": torch.rand(5, 3, 3, 6, 4, 4), "diagonal": 2},
    {"input": torch.rand(5, 3, 3, 6, 5, 5, 3), "diagonal": 1},
    # {"input": torch.rand(5, 3, 3, 6, 5, 5, 3, 4), "diagonal": 0},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_triu(input_data, data_type):
    comparator = testing.DefaultComparator(abs_diff=1e-2)
    test = testing.OpTest(
        func=torch.triu,
        input_args={
            "input": input_data["input"].to(data_type),
            "diagonal": input_data["diagonal"],
        },
        comparators=comparator,
    )
    test.check_result()
    test.check_musafp16_vs_musafp32()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_triu_(input_data, data_type):
    cpu_result = input_data["input"].to(data_type)
    cpu_result.triu(input_data["diagonal"])

    musa_result = input_data["input"].to(data_type).to("musa")
    musa_result.triu(input_data["diagonal"])

    assert testing.DefaultComparator()(cpu_result, musa_result.cpu())
    assert cpu_result.shape == musa_result.shape
    assert cpu_result.dtype == musa_result.dtype


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_triu_out(input_data, data_type):
    cpu_result = torch.zeros(input_data["input"].shape, dtype=data_type)
    torch.triu(
        input_data["input"].to(data_type), input_data["diagonal"], out=cpu_result
    )
    musa_result = torch.rand(input_data["input"].shape).to(data_type).to("musa")
    torch.triu(
        input_data["input"].to(data_type).to("musa"),
        input_data["diagonal"],
        out=musa_result,
    )

    assert testing.DefaultComparator()(cpu_result, musa_result.cpu())
    assert cpu_result.shape == musa_result.shape
    assert cpu_result.dtype == musa_result.dtype


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_tril(input_data, data_type):
    comparator = testing.DefaultComparator(abs_diff=1e-2)
    test = testing.OpTest(
        func=torch.tril,
        input_args={
            "input": input_data["input"].to(data_type),
            "diagonal": input_data["diagonal"],
        },
        comparators=comparator,
    )
    test.check_result()
    test.check_musafp16_vs_musafp32()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_tril_(input_data, data_type):
    cpu_result = input_data["input"].to(data_type)
    cpu_result.tril(input_data["diagonal"])

    musa_result = input_data["input"].to(data_type).to("musa")
    musa_result.tril(input_data["diagonal"])

    assert testing.DefaultComparator()(cpu_result, musa_result.cpu())
    assert cpu_result.shape == musa_result.shape
    assert cpu_result.dtype == musa_result.dtype


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_tril_out(input_data, data_type):
    cpu_result = torch.zeros(input_data["input"].shape, dtype=data_type)
    torch.tril(
        input_data["input"].to(data_type), input_data["diagonal"], out=cpu_result
    )
    musa_result = torch.rand(input_data["input"].shape).to(data_type).to("musa")
    torch.tril(
        input_data["input"].to(data_type).to("musa"),
        input_data["diagonal"],
        out=musa_result,
    )

    assert testing.DefaultComparator()(cpu_result, musa_result.cpu())
    assert cpu_result.shape == musa_result.shape
    assert cpu_result.dtype == musa_result.dtype
