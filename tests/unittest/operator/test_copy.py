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

input_data = [
    {"input": torch.randn(5, 3), "other": torch.randn(3)},
    {"input": torch.randn(10, 5, 3), "other": torch.randn(5, 3)},
    {"input": torch.randn(10, 5, 3), "other": torch.randn(3)},
    {"input": torch.randn(4096, 2048), "other": torch.randn(2048)},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", dtypes)
def test_copy_d2d(input_data, dtype):
    # gen cpu res
    dst = input_data["input"].clone().to(dtype)
    src = input_data["other"].clone().to(dtype)
    cpu_out = dst.copy_(src)

    # gen musa res
    musa_dst = input_data["input"].clone().musa().to(dtype)
    musa_src = input_data["other"].clone().musa().to(dtype)
    musa_out = musa_dst.copy_(musa_src)

    test = testing.OpTest(
        func=torch.copysign,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.compare_res(cpu_out, musa_out.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", dtypes)
def test_copy_h2d(input_data, dtype):
    # gen cpu res
    dst = input_data["input"].clone().to(dtype)
    src = input_data["other"].clone().to(dtype)
    cpu_out = dst.copy_(src)

    # gen musa res
    musa_dst = input_data["input"].clone().musa().to(dtype)
    musa_src = input_data["other"].clone().to(dtype)
    musa_out = musa_dst.copy_(musa_src)

    test = testing.OpTest(
        func=torch.copysign,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.compare_res(cpu_out, musa_out.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data", [{"input": torch.randn(10, 5, 3), "other": torch.randn(5)}]
)
@pytest.mark.parametrize("dtype", dtypes)
def test_copy_uncontiguous(input_data, dtype):
    # gen cpu res
    dst = input_data["input"].clone().to(dtype).select(-1, 0)
    src = input_data["other"].clone().to(dtype)
    cpu_out = dst.copy_(src)

    # gen musa res
    musa_dst = input_data["input"].clone().musa().to(dtype).select(-1, 0)
    musa_src = input_data["other"].clone().to(dtype)
    musa_out = musa_dst.copy_(musa_src)

    test = testing.OpTest(
        func=torch.copysign,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.compare_res(cpu_out, musa_out.cpu())
