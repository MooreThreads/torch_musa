"""Test unique operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

input_data = [
    {"input": torch.tensor(1.32)},
    {"input": torch.randn([1, 10])},
    {"input": torch.randn([1, 8, 4])},
    {"input": torch.randn([0, 10, 5, 5])},
    {
        "input": torch.randn([3, 2, 1, 1]).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn([9, 8, 7, 6, 5, 4]),
    },
]

dtypes = [
    torch.float32,
    torch.int32,
    torch.int64,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", dtypes)
@pytest.mark.parametrize("sort", [True])
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [True, False])
def test_unique(input_data, data_type, sort, return_inverse, return_counts):
    unit_input = {}
    if isinstance(input_data["input"], torch.Tensor):
        unit_input["input"] = input_data["input"].to(data_type)
    unit_input["return_inverse"] = return_inverse
    unit_input["return_counts"] = return_counts
    unit_input["sorted"] = sort
    test = testing.OpTest(
        func=torch.unique,
        input_args=unit_input,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", dtypes)
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [True, False])
def test_unique_consecutive(input_data, data_type, return_inverse, return_counts):
    unit_input = {}
    if isinstance(input_data["input"], torch.Tensor):
        unit_input["input"] = input_data["input"].to(data_type)
    # unit_input["return_inverse"] = return_inverse
    unit_input["return_counts"] = return_counts
    unit_input["return_inverse"] = return_inverse
    test = testing.OpTest(
        func=torch.unique_consecutive,
        input_args=unit_input,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    test.check_result()
    test.check_grad_fn()
