"""
Common functions.
"""

import os
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
    musa_backend = torch._C._get_privateuse1_backend_name()
    if device is None:
        if not torch_musa.is_available():
            raise ValueError("Moore Threads GPUs are not available")
        return torch_musa.current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
        if device.type == musa_backend and device.index is None:
            device = torch.device(device.type, torch_musa.current_device())
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in [musa_backend, "cpu"]:
                raise ValueError(f"Expected a musa or cpu device, but got: {device}")
        elif device.type != musa_backend:
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


def _get_musa_arch() -> int:
    """Get musa arch string, 21 for QY1, 22 for QY2, and so on."""
    user_defined_musa_arch = os.getenv("TORCH_MUSA_ARCH", None)
    if user_defined_musa_arch is None:
        try:
            properties = torch_musa.get_device_properties(0)
            major = properties.major
            minor = properties.minor
            musa_arch = int(major * 10 + minor)
            return musa_arch
        except Exception as err:  # pylint: disable=W0718
            print("get_devie_properties failed, reason:")
            print(err)
            print(
                "Users can set default musa arch with environment variable: "
                "'TORCH_MUSA_ARCH'"
            )
            print("Default musa arch properties is: 22")  # depend on CI machine
            return 22
    try:
        musa_arch = int(user_defined_musa_arch)
        print(f"Using pre-defined musa arch: {musa_arch}")
        assert musa_arch in [
            21,
            22,
            31,
        ], f"'TORCH_MUSA_ARCH' should be a string of int: 21, 22, or 31, got {musa_arch}"
        return musa_arch
    except Exception as err:  # pylint: disable=W0718
        print("'TORCH_MUSA_ARCH' should be a string of int: 21, 22, or 31, got ")
        print(f"  {musa_arch}\nwhich is not supported")
        raise ValueError from err
