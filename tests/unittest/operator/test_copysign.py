"""Test copysign operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
from torch_musa import testing


# Supported dtypes for copysign
dtypes = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn(3), "other": torch.randn(3)},
        {"input": torch.randn(10), "other": torch.randn(10)},
        {"input": torch.randn(20), "other": torch.randn(20)},
        {"input": torch.randn(20, 5), "other": torch.randn(20, 5)},
        {"input": torch.randn(3), "other": torch.tensor(1.0)},  # scalar case
        {"input": torch.tensor(1.0), "other": torch.randn(3)},  # scalar case
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
def test_copysign(input_data, dtype):
    # gen cpu res
    cpu_input = input_data["input"].clone().to(dtype)
    cpu_other = input_data["other"].clone().to(dtype)
    cpu_out = torch.copysign(cpu_input, cpu_other)

    # gen musa res
    musa_input = input_data["input"].clone().musa().to(dtype)
    musa_other = input_data["other"].clone().musa().to(dtype)
    musa_out = torch.copysign(musa_input, musa_other)

    test = testing.OpTest(
        func=torch.copysign,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.compare_res(cpu_out, musa_out.cpu())
