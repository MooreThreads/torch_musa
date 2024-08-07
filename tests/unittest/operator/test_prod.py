"""Test prod operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
from torch_musa import testing


def function(input_data, dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(func=func, input_args=input_data)
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([1, 10]), "dim": 1},
        {"input": torch.randn([1, 10, 5]), "dim": 2},
        {"input": torch.randn([1, 10, 5, 5]), "dim": 3},
        {
            "input": torch.randn([4, 10, 6, 5]).to(memory_format=torch.channels_last),
            "dim": 3,
        },
        {
            "input": torch.randn([4, 1, 6, 5]).to(memory_format=torch.channels_last),
            "dim": 3,
        },
        {
            "input": torch.randn([4, 2, 1, 1]).to(memory_format=torch.channels_last),
            "dim": 3,
        },
        {
            "input": torch.randn([0, 2, 1, 1]).to(memory_format=torch.channels_last),
            "dim": 3,
        },
        {"input": torch.randn([1, 10, 5, 5, 10]), "dim": 4},
        {"input": torch.randn([9, 8, 7, 6, 5, 4]), "dim": 5},
        {"input": torch.randn([9, 8, 7, 6, 5, 4, 3]), "dim": 6},
        {"input": torch.randn([0, 8, 7, 6, 5, 4, 3]), "dim": 6},
        {"input": torch.randn([9, 8, 7, 6, 5, 4, 5, 20]), "dim": 7},
        {"input": torch.randn([9, 8, 0, 6, 5, 4, 5, 20]), "dim": 7},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_prod(input_data, dtype):
    function(input_data, dtype, torch.prod)
