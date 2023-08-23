"""
Tensor attributes that are obtained from C++.
"""

import torch
import torch_musa

def _type(self, *args, **kwargs):
    """Get the data type of a musa tensor"""
    return torch_musa._MUSAC._type(self, *args, **kwargs)


@property
def _is_musa(self):
    """Check if a tensor is a musa tensor"""
    return torch_musa._MUSAC._is_musa(self)


def set_torch_attributes():
    """Set tensor attributes for torch musa."""
    torch.Tensor.type = _type
    torch.Tensor.is_musa = _is_musa
