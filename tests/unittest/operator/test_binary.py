"""Test binary operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {"input": torch.randn(10), "other": torch.randn(10)},
    {"input": torch.randn(10, 10), "other": torch.randn(10, 10)},
    {"input": torch.randn(10, 0), "other": torch.randn(10, 10)},
    {"input": torch.randn(10, 10, 2), "other": torch.randn(10, 10, 2)},
    {"input": torch.randn(10, 10, 2, 2), "other": torch.randn(10, 10, 2, 2)},
    {
        "input": torch.randn(10, 10, 2, 2, 1),
        "other": torch.randn(10, 10, 2, 2, 1),
    },
    {
        "input": torch.randn(10, 10, 2, 2, 1, 3),
        "other": torch.randn(10, 10, 2, 2, 1, 3),
    },
    {
        "input": torch.randn(10, 10, 2, 2, 1, 3, 2),
        "other": torch.randn(10, 10, 2, 2, 1, 3, 2),
    },
    {
        "input": torch.randn(10, 10, 2, 2, 1, 3, 2, 2),
        "other": torch.randn(10, 10, 2, 2, 1, 3, 2, 2),
    },
    {"input": torch.tensor(1.2), "other": torch.randn(30, 30)},
    {"input": torch.randn(30), "other": torch.tensor(1.2)},
    {"input": torch.randn(30, 1), "other": torch.randn(30, 30)},
    {"input": torch.randn(30, 1), "other": torch.randn(1, 30)},
]

other_scalar = [-9, 2.0, 5, 0, 12]


add_scalar_data_type = [torch.float32, torch.int32, torch.int64]


# test case like:  tensor([1]) + 5
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", add_scalar_data_type)
@pytest.mark.parametrize("other_scalar", other_scalar)
def test_add(input_data, data_type, other_scalar):
    cpu_result = input_data["input"].to(data_type) + other_scalar
    mtgpu_result = input_data["input"].to("musa", data_type) + other_scalar
    comparator = testing.DefaultComparator()
    assert cpu_result.shape == mtgpu_result.shape
    assert cpu_result.dtype == mtgpu_result.dtype
    assert comparator(mtgpu_result.cpu(), cpu_result)
