"""
Module attributes that are obtained from C++.
"""

from typing import Optional, Union, TypeVar
from torch.types import _device

import torch


T = TypeVar("T", bound="Module")


def _musa(self: T, device: Optional[Union[int, _device]] = None) -> T:
    r"""Moves all model parameters and buffers to the GPU.

    This also makes associated parameters and buffers different objects. So
    it should be called before constructing optimizer if the module will
    live on GPU while being optimized.

    .. note::
        This method modifies the module in-place.

    Args:
        device (int, optional): if specified, all parameters will be
            copied to that device

    Returns:
        Module: self
    """
    return self._apply(lambda t: t.musa(device))


def set_module_attributes():
    """Set module attributes for torch musa."""
    torch.nn.Module.musa = _musa
    torch.distributed.is_mccl_available = (
        torch.distributed.distributed_c10d.is_mccl_available
    )
