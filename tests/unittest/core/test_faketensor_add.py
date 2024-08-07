"""Test device features."""

# pylint: disable=invalid-name, comparison-with-itself, unused-variable, unused-import, C0303, C0114, C0116, R1703, R1705, W0611

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_musa


def validate_attributes(tensor_a, tensor_c):
    a_shape = tensor_a.shape
    a_dtype = tensor_a.dtype
    a_device = tensor_a.device

    c_shape = tensor_c.shape
    c_dtype = tensor_c.dtype
    c_device = tensor_c.device

    if c_shape == a_shape and c_dtype == a_dtype and c_device == a_device:
        return True
    else:
        return False


def test_faketensor_add():
    with FakeTensorMode():
        a = torch.tensor([1.2, 2.3], dtype=torch.float32, device="musa")
        b = torch.tensor([1.2, 2.3], dtype=torch.float32, device="cpu").to("musa")
        c = a + b

        assert validate_attributes(a, c)
