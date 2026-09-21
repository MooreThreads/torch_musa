"""
Tensor attributes that are obtained from C++.
"""

from typing import Optional, Union
from torch.types import _device

import torch
import torch_musa


def _musa(self, *args, **kwargs):
    """Returns a copy of this object in MUSA memory"""
    return torch_musa._MUSAC._musa(self, *args, **kwargs)


def _lazy_musa(self, *args, **kwargs):
    """Returns a copy of this object in MUSA memory"""
    return torch_musa._MUSAC._lazy_musa(self, *args, **kwargs)


def _pin_memory(self, device: Optional[Union[_device, str, None]] = "musa"):
    """Copies the tensor to pinned memory, if it’s not already pinned."""
    return self.orig_pin_memory(device)


def _is_pinned(self, device: Optional[Union[_device, str, None]] = "musa"):
    """Returns true if this tensor resides in pinned memory."""
    return self.orig_is_pinned(device)


@property
def _is_musa(self):
    """Check if a tensor is a musa tensor"""
    return torch_musa._MUSAC._is_musa(self)


def set_torch_attributes():
    """Set tensor attributes for torch musa."""
    torch.Tensor.is_musa = _is_musa
    torch.Tensor.musa = _musa
    torch.Tensor.lazy_musa = _lazy_musa
    # store original method
    torch.Tensor.orig_pin_memory = torch.Tensor.pin_memory
    torch.Tensor.orig_is_pinned = torch.Tensor.is_pinned
    # then we hack it with our customized function
    torch.Tensor.pin_memory = _pin_memory
    torch.Tensor.is_pinned = _is_pinned
