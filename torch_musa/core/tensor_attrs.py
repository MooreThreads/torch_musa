"""
Tensor attributes that are obtained from C++.
"""

import torch
import torch_musa

def _type(self, *args, **kwargs):
    """Get the data type of a musa tensor"""
    return torch_musa._MUSAC._type(self, *args, **kwargs)


def set_torch_attributes():
    """Set tensor attributes for torch musa."""
    torch.Tensor.type = _type
