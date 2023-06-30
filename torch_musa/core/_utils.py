"""
Common functions.
"""
from typing import Any, Union
import torch
from torch._utils import _get_device_index
import torch_musa

DeviceUnion = Union[torch.device, int, str, None]


def _get_musa_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    """Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.
    """
    if device is None:
        if not torch_musa.is_available():
            raise ValueError("Moore Threads GPUs are not available")
        return torch_musa.current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
        if device.type == "musa" and device.index is None:
            device = torch.device(device.type, torch_musa.current_device())
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


def _dummy_type(name: str) -> type:
    """Define dummy class named 'name'.The class can't instantiate."""

    def get_err_fn(is_init: bool):
        def err_fn(obj):
            if is_init:
                class_name = obj.__class__.__name__
            else:
                class_name = obj.__name__
            raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")

        return err_fn

    return type(
        name, (object,), {"__init__": get_err_fn(True), "__new__": get_err_fn(False)}
    )
