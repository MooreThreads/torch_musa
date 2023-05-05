"""MUSA model serialization features."""
from typing import Any
import torch
import torch_musa


_REGISTRATION_KEY = 999


def _validate_musa_device(location: str) -> None:
    """Validate the musa device location."""
    device_idx = (
        torch._utils._get_device_index(
            torch.device(location), optional=True, allow_cpu=False
        )
        or 0
    )

    if not torch_musa.is_available():
        raise RuntimeError(
            "Attempting to deserialize object on a MUSA "
            "device but torch_musa.device.is_available() is False. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    device_count = torch_musa.device_count()
    if device_idx >= device_count:
        raise RuntimeError(
            "Attempting to deserialize object on MUSA device "
            f"{device_idx} but torch_musa.device.device_count() "
            f"is {device_count}. Please use torch.load with map_location "
            "to map your storages to an existing device."
        )


def _musa_tag(obj: Any) -> Any:
    """Get musa device tag."""
    if obj.device.type == "musa":
        return "musa:" + str(obj.device.index)
    return None


def _musa_deserialize(obj: Any, location: str) -> Any:
    """Musa device deserialization."""
    if location.startswith("musa"):
        _validate_musa_device(location)
        if obj.device.type != "cpu":
            # Return a MTGPU copy of this storage if it's not already on the MTGPU.
            return torch.UntypedStorage(
                obj.size(), device=torch.device(location)
            ).copy_(obj, False)
        return obj
    return None


def register_deserialization() -> None:
    """Register musa deserialization."""
    torch.serialization.register_package(
        _REGISTRATION_KEY, _musa_tag, _musa_deserialize
    )
