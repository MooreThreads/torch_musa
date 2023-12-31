"""Unittest for memory APIs."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, unused-variable, unexpected-keyword-arg
import pytest
import torch
from torch import nn

import torch_musa
from torch_musa import testing


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
