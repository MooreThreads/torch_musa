"""Test tensor type"""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

torch_dot_musa_tensor_types = [
    "torch.musa.ByteTensor",
    "torch.musa.ShortTensor",
    "torch.musa.IntTensor",
    "torch.musa.LongTensor",
    "torch.musa.FloatTensor",
    "torch.musa.DoubleTensor",
    "torch.musa.BoolTensor",
    torch.musa.ByteTensor,
    torch.musa.ShortTensor,
    torch.musa.IntTensor,
    torch.musa.LongTensor,
    torch.musa.FloatTensor,
    torch.musa.DoubleTensor,
    torch.musa.BoolTensor,
]

torch_musa_tensor_types = [
    "torch_musa.ByteTensor",
    "torch_musa.ShortTensor",
    "torch_musa.IntTensor",
    "torch_musa.LongTensor",
    "torch_musa.FloatTensor",
    "torch_musa.DoubleTensor",
    "torch_musa.BoolTensor",
    torch_musa.ByteTensor,
    torch_musa.ShortTensor,
    torch_musa.IntTensor,
    torch_musa.LongTensor,
    torch_musa.FloatTensor,
    torch_musa.DoubleTensor,
    torch_musa.BoolTensor,
]

torch_tensor_types = [
    "torch.ByteTensor",
    "torch.ShortTensor",
    "torch.IntTensor",
    "torch.LongTensor",
    "torch.FloatTensor",
    "torch.DoubleTensor",
    "torch.BoolTensor",
    torch.ByteTensor,
    torch.ShortTensor,
    torch.IntTensor,
    torch.LongTensor,
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.BoolTensor,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_tensor_type():
    for torch_dot_musa_tensor_type, torch_musa_tensor_type, torch_tensor_type in zip(
        torch_dot_musa_tensor_types, torch_musa_tensor_types, torch_tensor_types
    ):
        torch_dot_musa_tensor = torch.arange(64).type(torch_dot_musa_tensor_type)
        torch_musa_tensor = torch.arange(64).type(torch_musa_tensor_type)
        torch_tensor = torch.arange(64).type(torch_tensor_type)
        assert torch.allclose(torch_dot_musa_tensor.cpu(), torch_musa_tensor.cpu())
        assert torch.allclose(torch_dot_musa_tensor.cpu(), torch_tensor)
