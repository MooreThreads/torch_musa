"""
Module attributes that are obtained from C.
"""

# pylint: disable=C0301, C0103, R1720

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


def _lazy_to(self, *args, **kwargs):
    r"""Lazy version of :meth:`~Module.to` for use with unified memory.

    Same signature and behavior as :meth:`~Module.to`, but when called
    within a unified memory allocator context, device transfers only
    relabel the device metadata without copying data.

    .. warning::
        Must be called within ``torch_musa.use_unified_allocator()`` or
        ``torch_musa.use_unified_cpu_allocator()`` context manager.
        Using outside these contexts will raise an error.

    Returns:
        Module: self
    """
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
        *args, **kwargs
    )

    if dtype is not None:
        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError(
                "nn.Module.to only accepts floating point or complex "
                f"dtypes, but got desired dtype={dtype}"
            )
        if dtype.is_complex:
            warnings.warn(
                "Complex modules are a new feature under active development whose design may change, "
                "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                "if a complex module does not work as expected."
            )

    def convert(t):
        try:
            if convert_to_format is not None and t.dim() in (4, 5):
                return t.lazy_to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                    memory_format=convert_to_format,
                )
            return t.lazy_to(
                device,
                dtype if t.is_floating_point() or t.is_complex() else None,
                non_blocking,
            )
        except NotImplementedError as e:
            if str(e) == "Cannot copy out of meta tensor; no data!":
                raise NotImplementedError(
                    f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                    f"when moving module from meta to a different device."
                ) from None
            else:
                raise

    return self._apply(convert)


def set_module_attributes():
    """Set module attributes for torch musa."""
    torch.nn.Module.musa = _musa
    torch.nn.Module.lazy_to = _lazy_to
    torch.distributed.is_mccl_available = (
        torch.distributed.distributed_c10d.is_mccl_available
    )
