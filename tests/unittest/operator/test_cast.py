"""Test cast operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import numpy as np
import torch
import pytest
import torch_musa

from torch_musa import testing


input_data = testing.get_raw_data()
all_dtypes = [
    torch.bool,
    torch.uint8,
    torch.float32,
    torch.int32,
    torch.float64,
    torch.int64
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("src_dtype", all_dtypes)
def test_cast_to_float64(input_data, src_dtype):
    call_cast_func(input_data, src_dtype, torch.float64)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("src_dtype", all_dtypes)
def test_cast_to_float32(input_data, src_dtype):
    call_cast_func(input_data, src_dtype, torch.float32)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("src_dtype", all_dtypes)
def test_cast_to_int64(input_data, src_dtype):
    call_cast_func(input_data, src_dtype, torch.int64)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("src_dtype", all_dtypes)
def test_cast_to_int32(input_data, src_dtype):
    call_cast_func(input_data, src_dtype, torch.int32)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("src_dtype", all_dtypes)
def test_cast_to_uint8(input_data, src_dtype):
    call_cast_func(input_data, src_dtype, torch.uint8)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("src_dtype", all_dtypes)
def test_cast_to_bool(input_data, src_dtype):
    call_cast_func(input_data, src_dtype, torch.bool)


def call_cast_func(input_data, src_dtype, dst_dtype):
    # this cast will call Permute op or memcpy func
    if src_dtype == dst_dtype:
        return
    src_tensor_mtgpu = input_data.clone().detach().to(dtype=src_dtype, device="musa")
    dst_tensor_mtgpu = src_tensor_mtgpu.to(dst_dtype)
    mtgpu_result = dst_tensor_mtgpu.cpu().detach()

    src_tensor_cpu = input_data.clone().detach().to(dtype=src_dtype, device="cpu")
    dst_tensor_cpu = src_tensor_cpu.to(dst_dtype)
    cpu_result = dst_tensor_cpu.detach()

    comparator = testing.DefaultComparator()
    assert comparator(mtgpu_result, cpu_result)
    assert mtgpu_result.dtype == cpu_result.dtype
    assert mtgpu_result.shape == cpu_result.shape
