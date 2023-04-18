"""Unittest for memory APIs."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
from torch import nn

import torch_musa


def  test_single_op():
    """Test a single operator"""
    torch_musa.empty_cache()
    input_data = torch.randn(1024, 1024).to("musa")
    torch.matmul(input_data, input_data)

    m_allocated = torch_musa.max_memory_allocated()
    m_reserved = torch_musa.max_memory_reserved()
    allocated = torch_musa.memory_allocated()
    reserved = torch_musa.memory_reserved()
    assert m_allocated == 8 * 1024 * 1024 # maximally allocated 8MB
    assert m_reserved == 20 * 1024 * 1024 # maximally reserved 20MB
    assert allocated == 4 * 1024 * 1024 # allocated 4MB
    assert reserved == 20 * 1024 * 1024 # reserved 20MB


def test_multiple_ops():
    """Test multiple ops"""
    torch_musa.empty_cache()
    input_data = torch.randn(1024 * 4, 1024 * 4).to("musa") # 16 * 1024 * 1024 * 4 = 64MB
    tensor_a = torch.matmul(input_data, input_data)
    torch.matmul(tensor_a, tensor_a)
    summary = torch_musa.memory_summary()
    assert "reserved" in summary

    stats = torch_musa.memory_stats()
    assert isinstance(stats, dict)

    m_allocated = torch_musa.max_memory_allocated()
    m_reserved = torch_musa.max_memory_reserved()
    allocated = torch_musa.memory_allocated()
    reserved = torch_musa.memory_reserved()
    assert m_allocated == 192 * 1024 * 1024 # maximally allocated 192MB
    assert m_reserved == 192 * 1024 * 1024 # maximally reserved 192MB
    assert allocated == 128 * 1024 * 1024 # allocated 128MB
    assert reserved == 192 * 1024 * 1024 # reserved 192MB
