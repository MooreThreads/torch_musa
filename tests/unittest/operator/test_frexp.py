"""Test frexp operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(0),
    },
    {
        "input": torch.randn(10),
    },
    {
        "input": torch.randn(4, 9),
    },
    {
        "input": torch.randn(100, 30),
    },
    {
        "input": torch.randn(2, 3, 4),
    },
    {
        "input": torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                2.0,
                0.5,
                1024.0,
                65504.0,
                1e-7,
                float("inf"),
                float("-inf"),
                float("nan"),
            ]
        )
    },
]

input_dtype = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", input_dtype)
def test_frexp(input_data, dtype):
    cmp = testing.DefaultComparator()

    def fwd(device="cpu"):
        x = input_data["input"].to(device).to(dtype)
        mantissa, exponent = torch.frexp(x)

        return mantissa, exponent

    cpu_mantissa, cpu_exponent = fwd("cpu")
    musa_mantissa, musa_exponent = fwd("musa")

    cmp(cpu_mantissa, musa_mantissa.cpu())
    cmp(cpu_exponent, musa_exponent.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", input_dtype)
def test_frexp_out(input_data, dtype):
    cmp = testing.DefaultComparator()

    def fwd(device="cpu"):
        x = input_data["input"].to(device).to(dtype)
        mantissa = torch.empty_like(x)
        exponent = torch.empty_like(x, dtype=torch.int32)
        torch.frexp(x, out=[mantissa, exponent])

        return mantissa, exponent

    cpu_mantissa, cpu_exponent = fwd("cpu")
    musa_mantissa, musa_exponent = fwd("musa")

    cmp(cpu_mantissa, musa_mantissa.cpu())
    cmp(cpu_exponent, musa_exponent.cpu())
