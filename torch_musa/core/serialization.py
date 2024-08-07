"""MUSA model serialization features."""

from typing import Any
import torch


_REGISTRATION_KEY = 999


def _validate_musa_device(location: str) -> None:
    """Validate the musa device location."""
    device_idx = (
        torch._utils._get_device_index(
            torch.device(location), optional=True, allow_cpu=False
        )
        or 0
    )

    if not torch.musa.is_available():
        raise RuntimeError(
            "Attempting to deserialize object on a MUSA "
            "device but torch.musa.device.is_available() is False. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    device_count = torch.musa.device_count()
    if device_idx >= device_count:
        raise RuntimeError(
            "Attempting to deserialize object on MUSA device "
            f"{device_idx} but torch.musa.device.device_count() "
            f"is {device_count}. Please use torch.load with map_location "
            "to map your storages to an existing device."
        )
    return device_idx


def _musa_tag(obj: Any) -> Any:
    """Get musa device tag."""
    if obj.device.type == "musa":
        return "musa:" + str(obj.device.index)
    return None


def _musa_deserialize(obj: Any, location: str) -> Any:
    """Musa device deserialization."""
    if location.startswith("musa"):
        device = _validate_musa_device(location)
        if getattr(obj, "_torch_load_uninitialized", False):
            with torch.musa.device(device):
                return torch.UntypedStorage(obj.nbytes(), device=torch.device(location))
        else:
            return obj.musa(device)
    return None


def _cuda_deserialize(_: Any, location: str) -> Any:
    """Deserialization function for loading serialized models which tagged with cuda"""
    if location.startswith("cuda"):
        raise RuntimeError(
            "Attempting to deserialize object on a CUDA "
            "device but torch.cuda.is_available() is False. "
            "If you are running on a MUSA enabled machine, "
            "please use torch.load with map_location=torch.device('musa') "
            "or map_location=torch.device('cpu') "
            "to map your storages to the MUSA or CPU."
        )


def register_deserialization() -> None:
    """Register musa deserialization."""
    torch.serialization.register_package(
        _REGISTRATION_KEY, _musa_tag, _musa_deserialize
    )

    # DONT want to add patch to torch, so just substitute `_cuda_deserialize` here
    _cuda_idx = 1
    _registeration_key, _cuda_tag, _ = torch.serialization._package_registry[_cuda_idx]
    torch.serialization._package_registry.pop(_cuda_idx)
    torch.serialization.register_package(
        _registeration_key, _cuda_tag, _cuda_deserialize
    )
    torch.serialization._package_registry.sort()
