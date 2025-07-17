"""Test functionality of storage resize"""

import torch

from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_untyped_storage_resize():
    """test untyped storage resize_ both on CPU and MUSA"""
    x_cpu = torch.randn((1024,))
    x_musa = torch.randn((1024,), device="musa")

    new_size = 2048
    out_cpu = x_cpu.untyped_storage().resize_(new_size)
    out_musa = x_musa.untyped_storage().resize_(new_size)
    assert out_cpu.size() == out_musa.size() and out_musa.size() == new_size
    assert out_cpu.device.type == "cpu" and out_musa.device.type == "musa"

    new_size = 0
    x_cpu.untyped_storage().resize_(new_size)
    x_musa.untyped_storage().resize_(new_size)
    assert out_cpu.size() == out_musa.size() and out_musa.size() == 0
    assert out_cpu.device.type == "cpu" and out_musa.device.type == "musa"
