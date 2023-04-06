"""
This package adds support for Moore Threads GPU device type implementation.
"""

from typing import Any
from functools import lru_cache
from torch._utils import _get_device_index
import torch
import torch_musa


def _get_musa_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    """Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["musa", "cpu"]:
                raise ValueError(f"Expected a musa or cpu device, but got: {device}")
        elif device.type != "musa":
            raise ValueError(f"Expected a musa device, but got: {device}")
    if not torch.jit.is_scripting():
        if isinstance(device, torch_musa.device):
            return device.idx
    return _get_device_index(device, optional, allow_cpu)


class Device(object):
    """Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        # get specified device index
        self.idx = _get_musa_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return

        self.prev_idx = torch_musa.current_device()
        if self.idx != self.prev_idx:
            torch_musa.set_device(self.idx)

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch_musa.set_device(self.prev_idx)
        return False


def set_device(device: Any) -> None:
    """Sets the current device.

    In most cases it's better to user ``MUSA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_musa_device_index(device)
    if device >= 0:
        torch_musa._MUSAC._musa_setDevice(device)


def current_device() -> int:
    """Returns the index of a currently selected device."""
    return torch_musa._MUSAC._musa_getDevice()


def is_available() -> bool:
    """Returns a bool indicating if MUSA is currently available."""
    return torch_musa._MUSAC._musa_getDeviceCount() > 0


def synchronize(device: Any) -> None:
    """Waits for all kernels in all streams on a MUSA device to complete."""
    with torch_musa.device(device):
        return torch_musa._MUSAC._musa_synchronize()


@lru_cache(maxsize=1)
def device_count() -> int:
    """Returns the number of Moore Threads GPUs avaiable."""
    return torch_musa._MUSAC._musa_getDeviceCount()
