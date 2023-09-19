"""Unittest for memory APIs."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, unused-variable, unexpected-keyword-arg
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torch_musa
from torch_musa import testing
from torch_musa.testing import get_cycles_per_ms


def test_single_op():
    """Test a single operator"""
    torch.musa.empty_cache()
    torch.musa.reset_peak_stats()
    input_data = torch.randn(1024, 1024).to("musa")
    torch.matmul(input_data, input_data)

    m_allocated = torch.musa.max_memory_allocated()
    m_reserved = torch.musa.max_memory_reserved()
    allocated = torch.musa.memory_allocated()
    reserved = torch.musa.memory_reserved()
    assert m_allocated == 8 * 1024 * 1024  # maximally allocated 8MB
    assert m_reserved == 20 * 1024 * 1024  # maximally reserved 20MB
    assert allocated == 4 * 1024 * 1024  # allocated 4MB
    assert reserved == 20 * 1024 * 1024  # reserved 20MB


def test_multiple_ops():
    """Test multiple ops"""
    torch.musa.empty_cache()
    torch.musa.reset_peak_stats()
    input_data = torch.randn(1024 * 4, 1024 * 4).to("musa")  # 16 * 1024 * 1024 * 4 = 64MB
    tensor_a = torch.matmul(input_data, input_data)
    torch.matmul(tensor_a, tensor_a)
    summary = torch.musa.memory_summary()
    assert "reserved" in summary

    stats = torch.musa.memory_stats()
    assert isinstance(stats, dict)

    m_allocated = torch.musa.max_memory_allocated()
    m_reserved = torch.musa.max_memory_reserved()
    allocated = torch.musa.memory_allocated()
    reserved = torch.musa.memory_reserved()
    assert m_allocated == 192 * 1024 * 1024  # maximally allocated 192MB
    assert m_reserved == 192 * 1024 * 1024  # maximally reserved 192MB
    assert allocated == 128 * 1024 * 1024  # allocated 128MB
    assert reserved == 192 * 1024 * 1024  # reserved 192MB


def test_single_device_ops():
    """Test memory stats ops when specify single device"""
    torch.musa.empty_cache()
    torch.musa.reset_peak_stats()
    input_data_small = torch.randn(1024 * 4, 1024 * 4).to("musa:0")  # 16 * 1024 * 1024 * 4 = 64MB
    input_data_large = torch.randn(1024 * 8, 1024 * 8).to("musa:0")  # 64 * 1024 * 1024 * 4 = 256MB
    del input_data_large  # free 256MB on device 0
    summary = torch.musa.memory_summary(0)
    assert "reserved" in summary

    stats = torch.musa.memory_stats(0)
    assert isinstance(stats, dict)

    m_allocated = torch.musa.max_memory_allocated(0)
    m_reserved = torch.musa.max_memory_reserved(0)
    allocated = torch.musa.memory_allocated(0)
    reserved = torch.musa.memory_reserved(0)
    assert m_allocated == 320 * 1024 * 1024  # maximally allocated 320MB
    assert m_reserved == 320 * 1024 * 1024  # maximally reserved 320MB
    assert allocated == 64 * 1024 * 1024  # allocated 64MB
    assert reserved == 320 * 1024 * 1024  # reserved 320MB


@testing.skip_if_not_multiple_musa_device
def test_all_devices_ops():
    """Test memory stats ops when specify all devices"""
    torch.musa.empty_cache()
    torch.musa.reset_peak_stats()
    tensor_0 = torch.randn(1024 * 4, 1024 * 4).to("musa:0")  # 16 * 1024 * 1024 * 4 = 64MB
    tensor_1 = torch.randn(1024 * 8, 1024 * 8).to("musa:1")  # 64 * 1024 * 1024 * 4 = 256MB
    summary = torch.musa.memory_summary(all_device=True)
    assert "reserved" in summary
    assert "[ALL]" in summary

    stats = torch.musa.memory_stats_all()
    assert isinstance(stats, dict)

    m_allocated = torch.musa.max_memory_allocated(all_device=True)
    m_reserved = torch.musa.max_memory_reserved(all_device=True)
    allocated = torch.musa.memory_allocated(all_device=True)
    reserved = torch.musa.memory_reserved(all_device=True)
    assert m_allocated == 256 * 1024 * 1024  # maximally allocated 256MB
    assert m_reserved == 256 * 1024 * 1024  # maximally reserved 256MB
    assert allocated == (64 + 256) * 1024 * 1024  # allocated 320MB
    assert reserved == (64 + 256) * 1024 * 1024  # reserved 320MB


def test_caching_pinned_memory():
    """Test caching host allocator functions."""
    # check that allocations are re-used after deletion
    pinned_tensor = torch.tensor([1]).pin_memory("musa")
    ptr = pinned_tensor.data_ptr()
    del pinned_tensor
    pinned_tensor = torch.tensor([1]).pin_memory("musa")
    assert pinned_tensor.data_ptr() == ptr, "allocation not reused."

    # check that the allocation is not re-used if it's in-use by a copy
    gpu_tensor = torch.tensor([0], device="musa")
    torch.musa._sleep(int(1000 * get_cycles_per_ms()))  # delay the copy by 1s
    gpu_tensor.copy_(pinned_tensor, non_blocking=True)
    del pinned_tensor
    pinned_tensor = torch.tensor([1]).pin_memory("musa")
    assert pinned_tensor.data_ptr() != ptr, "allocation re-used too soon."
    assert list(gpu_tensor) == [1]

    # check pin memory copy with different dtypes
    gpu_tensor = torch.tensor([0.0], device="musa", dtype=torch.float32)
    pinned_tensor = torch.tensor([1], dtype=torch.int8).pin_memory("musa")
    gpu_tensor.copy_(pinned_tensor, non_blocking=True)
    assert list(gpu_tensor) == [1]

    # check pin memory D2H copy
    gpu_tensor = torch.tensor([1], device="musa")
    pinned_tensor = torch.tensor([0], dtype=torch.int32).pin_memory("musa")
    pinned_tensor.copy_(gpu_tensor)
    assert list(pinned_tensor) == [1]


class DictDataset(Dataset):
    """Dictionary data loader."""
    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return {
            "a_tensor": torch.empty(4, 2).fill_(ndx),
            "another_dict": {
                "a_number": ndx,
            },
        }


def test_pin_memory_dataloader():
    """Test dataloader with pin memory."""
    dataset = DictDataset()
    loader = DataLoader(dataset, batch_size=2, pin_memory=True, pin_memory_device="musa")
    for sample in loader:
        assert sample["a_tensor"].is_pinned("musa")
        assert sample["another_dict"]["a_number"].is_pinned("musa")


@testing.skip_if_not_multiple_musa_device
def test_pin_memory_dataloader_non_zero_device():
    """Test dataloader with pin memory on non-zero gpu."""
    dataset = DictDataset()
    loader = DataLoader(dataset, batch_size=2, pin_memory=True, pin_memory_device="musa:1")
    for sample in loader:
        assert sample["a_tensor"].is_pinned("musa:1")
        assert sample["another_dict"]["a_number"].is_pinned("musa:1")
