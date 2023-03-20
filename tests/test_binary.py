"""Test binary operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import numpy as np
import pytest

from base_test_tool import DefaultComparator

import torch_musa

input_data = [
    {"input": np.random.randn(10), "other": np.random.randn(10)},
    {"input": np.random.randn(10, 10), "other": np.random.randn(10, 10)},
    {"input": np.random.randn(10, 0), "other": np.random.randn(10, 10)},
    {"input": np.random.randn(10, 10, 2), "other": np.random.randn(10, 10, 2)},
    {"input": np.random.randn(10, 10, 2, 2), "other": np.random.randn(10, 10, 2, 2)},
    {
        "input": np.random.randn(10, 10, 2, 2, 1),
        "other": np.random.randn(10, 10, 2, 2, 1),
    },
    {
        "input": np.random.randn(10, 10, 2, 2, 1, 3),
        "other": np.random.randn(10, 10, 2, 2, 1, 3),
    },
    {
        "input": np.random.randn(10, 10, 2, 2, 1, 3, 2),
        "other": np.random.randn(10, 10, 2, 2, 1, 3, 2),
    },
    {
        "input": np.random.randn(10, 10, 2, 2, 1, 3, 2, 2),
        "other": np.random.randn(10, 10, 2, 2, 1, 3, 2, 2),
    },
    {"input": np.array(1.2), "other": np.random.randn(30, 30)},
    {"input": np.random.randn(30), "other": np.array(1.2)},
    {"input": np.random.randn(30, 1), "other": np.random.randn(30, 30)},
    {"input": np.random.randn(30, 1), "other": np.random.randn(1, 30)},
]

other_scalar = [-9, 2.0, 5, 0, 12]


add_scalar_data_type = [np.float32, np.int32, np.int64]


# test case like:  tensor([1]) + 5
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", add_scalar_data_type)
@pytest.mark.parametrize("other_scalar", other_scalar)
def test_add(input_data, data_type, other_scalar):
    cpu_result = (
        torch.tensor(input_data["input"].astype(data_type), device="cpu") + other_scalar
    )
    cpu_result_numpy = cpu_result.detach().numpy()
    mtgpu_result = (
        torch.tensor(input_data["input"].astype(data_type), device="mtgpu")
        + other_scalar
    )
    mtgpu_result_numpy = mtgpu_result.cpu().detach().numpy()
    comparator = DefaultComparator()
    assert cpu_result.shape == mtgpu_result.shape
    assert cpu_result.dtype == mtgpu_result.dtype
    assert comparator(mtgpu_result_numpy, cpu_result_numpy)
