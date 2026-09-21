"""Public MUSA IPC helpers.

This module exposes a small Python facade over Torch-MUSA's existing
storage-sharing internals. Raw storage reduction tuples remain an
implementation detail.
"""

# pylint: disable=protected-access

from __future__ import annotations
import base64
from functools import lru_cache
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import torch
import torch_musa

from ._utils import DeviceUnion as _device_t


__all__ = [
    "MUSAIPCTensorHandle",
    "MUSAIPCTensor",
    "export_tensor",
    "open_tensor",
]


_musa_device = torch._C._get_privateuse1_backend_name()
_STORAGE_BYTES_FIELDS = (1, 4, 6)


@dataclass(frozen=True, repr=False)
class MUSAIPCTensorHandle:
    """Transport metadata for an exported MUSA tensor."""

    shared_storage_info: Tuple[Any, ...]
    dtype: torch.dtype
    size: Tuple[int, ...]
    stride: Tuple[int, ...]
    storage_offset: int
    requires_grad: bool
    device_uuid: str

    @staticmethod
    def _format_blob(value: bytes) -> str:
        return f"len={len(value)}, hex={value.hex()}"

    def _format_shared_storage_info(self) -> str:
        shared_storage_info = self.shared_storage_info
        if len(shared_storage_info) != 8:
            return repr(shared_storage_info)

        shared_storage_field_names = (
            "device_index",
            "storage_handle",
            "storage_size",
            "storage_offset",
            "ref_counter_handle",
            "ref_counter_offset",
            "event_handle",
            "event_sync_required",
        )
        field_reprs = []
        for name, value in zip(shared_storage_field_names, shared_storage_info):
            value_repr = (
                repr(self._format_blob(value))
                if isinstance(value, bytes)
                else repr(value)
            )
            field_reprs.append(f"{name}={value_repr}")
        return f"({', '.join(field_reprs)})"

    def to_dict(self) -> Dict[str, Any]:
        shared_storage_info = list(self.shared_storage_info)
        for field_index in _STORAGE_BYTES_FIELDS:
            shared_storage_info[field_index] = base64.b64encode(
                shared_storage_info[field_index]
            ).decode("ascii")
        return {
            "shared_storage_info": shared_storage_info,
            "dtype": str(self.dtype),
            "size": list(self.size),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "requires_grad": self.requires_grad,
            "device_uuid": self.device_uuid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MUSAIPCTensorHandle:
        return cls(
            shared_storage_info=tuple(
                base64.b64decode(value) if index in _STORAGE_BYTES_FIELDS else value
                for index, value in enumerate(data["shared_storage_info"])
            ),
            dtype=getattr(torch, data["dtype"][len("torch.") :]),
            size=tuple(data["size"]),
            stride=tuple(data["stride"]),
            storage_offset=data["storage_offset"],
            requires_grad=data["requires_grad"],
            device_uuid=data["device_uuid"],
        )

    def __repr__(self) -> str:
        field_reprs = []
        for data_field in fields(self):
            value = getattr(self, data_field.name)
            value_repr = (
                self._format_shared_storage_info()
                if data_field.name == "shared_storage_info"
                else repr(value)
            )
            field_reprs.append(f"{data_field.name}={value_repr}")
        return f"{type(self).__qualname__}({', '.join(field_reprs)})"

    __str__ = __repr__


class MUSAIPCTensor:
    """Owner object for a receiver-side imported MUSA tensor."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor: Optional[torch.Tensor] = tensor

    @property
    def tensor(self) -> torch.Tensor:
        tensor = self._tensor
        if tensor is None:
            raise RuntimeError("MUSA IPC tensor owner is closed")
        return tensor

    @property
    def closed(self) -> bool:
        return self._tensor is None

    def close(self) -> None:
        self._tensor = None

    def __del__(self) -> None:
        try:
            self.close()
        # pylint: disable-next=broad-exception-caught
        except Exception:  # pragma: no cover - destructors must not raise
            pass

    def __enter__(self) -> MUSAIPCTensor:
        if self.closed:
            raise RuntimeError("MUSA IPC tensor owner is closed")
        return self

    def __exit__(
        self, _exc_type, _exc, _traceback
    ) -> None:  # type: ignore[no-untyped-def]
        self.close()


def _get_device_uuid(device: _device_t = None) -> str:
    """Return the UUID for a visible MUSA device."""
    return str(torch_musa.get_device_properties(device).uuid)


@lru_cache(maxsize=1)
def _visible_device_uuids() -> Dict[int, str]:
    if not torch_musa.is_available():
        raise RuntimeError("Cannot open MUSA IPC handle: MUSA is not available")
    return {
        device_index: _get_device_uuid(device_index)
        for device_index in range(torch_musa.device_count())
    }


@lru_cache(maxsize=None)
def _resolve_device(device_uuid: str) -> int:
    visible_uuids = _visible_device_uuids()
    for local_index, local_uuid in visible_uuids.items():
        if local_uuid == device_uuid:
            return local_index
    raise RuntimeError(
        "Could not resolve MUSA IPC handle device UUID "
        f"{device_uuid!r}. visible receiver UUIDs are {visible_uuids}"
    )


def export_tensor(tensor: torch.Tensor) -> MUSAIPCTensorHandle:
    """Export a MUSA tensor view as an IPC handle."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.device.type != _musa_device:
        raise RuntimeError("MUSA IPC tensor export requires a MUSA tensor")

    return MUSAIPCTensorHandle(
        shared_storage_info=tensor.untyped_storage()._share_musa_(),
        dtype=tensor.dtype,
        size=tuple(tensor.size()),
        stride=tuple(tensor.stride()),
        storage_offset=tensor.storage_offset(),
        requires_grad=tensor.requires_grad,
        device_uuid=_get_device_uuid(tensor.device),
    )


def _open_storage(handle: MUSAIPCTensorHandle) -> torch.UntypedStorage:
    """Open storage through Torch-MUSA's private sharing bridge."""
    shared_storage_info = handle.shared_storage_info
    if len(shared_storage_info) != 8:
        raise ValueError("MUSA shared storage metadata must contain 8 items")

    device_index = _resolve_device(handle.device_uuid)

    # Replace the producer-local device index with the receiver-local index.
    return torch.UntypedStorage._new_shared_musa(device_index, *shared_storage_info[1:])


def open_tensor(handle: MUSAIPCTensorHandle) -> MUSAIPCTensor:
    """Open a MUSA IPC tensor handle and return an owner object."""

    storage = _open_storage(handle)
    return MUSAIPCTensor(
        torch.empty(0, device=storage.device, dtype=handle.dtype)
        .set_(storage, handle.storage_offset, handle.size, handle.stride)
        .requires_grad_(handle.requires_grad)
    )
