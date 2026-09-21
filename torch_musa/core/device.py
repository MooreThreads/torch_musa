"""
This package adds support for Moore Threads GPU device type implementation.
"""

# pylint: disable=W0622, W0718


from typing import Any, Tuple, Optional, List
from functools import lru_cache
import torch
import torch_musa._MUSAC
from ._lazy_init import _lazy_init
from ._utils import (
    _get_musa_device_index,
    DeviceUnion as _device_t,
)

_MusaDeviceProperties = torch_musa._MUSAC._MusaDeviceProperties
_exchange_device = torch_musa._MUSAC._musa_exchangeDevice
_maybe_exchange_device = torch_musa._MUSAC._musa_maybeExchangeDevice


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
        self.prev_idx = torch_musa._exchange_device(self.idx)

    def __exit__(self, *args):
        self.idx = torch_musa._maybe_exchange_device(self.prev_idx)
        return False


class DeviceOf(Device):
    """Context-manager that changes the current device to that of given object.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        musa_backend = torch._C._get_privateuse1_backend_name()
        idx = obj.get_device() if obj.device.type == musa_backend else -1
        super().__init__(idx)


def set_device(device: _device_t) -> None:
    """Sets the current device.

    In most cases it's better to user ``MUSA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_musa_device_index(device)
    if device >= 0:
        torch_musa._MUSAC._musa_setDevice(device)


def get_device_name(device: Optional[_device_t] = None) -> str:
    """Gets the name of device.

    Args:
        device (torch.device, torch_musa.device or int).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
    """Gets the musa capability of a device.

    Args:
        device (torch.device, torch_musa.Device, int or string, optional): device for
            which to return the device capability.

    Returns:
        tuple(int, int): the major and minor musa capability of the device.
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_properties(device: Optional[_device_t] = None) -> _MusaDeviceProperties:
    """Gets the properties of a device.

    Args:
        device (torch.device, torch_musa.device or int): device for which
        to return the properties of the device.

    Returns:
        _MusaDeviceProperties: the properties of the device.
    """
    _lazy_init()
    device = _get_musa_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return torch_musa._get_device_properties(device)


def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool:
    """Checks if peer access between two device is possible."""
    _lazy_init()
    device = _get_musa_device_index(device, optional=True)
    peer_device = _get_musa_device_index(peer_device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    if peer_device < 0 or peer_device >= device_count():
        raise AssertionError("Invalid peer device id")
    return torch_musa._MUSAC._musa_canDeviceAccessPeer(device, peer_device)


def current_device() -> int:
    """Returns the index of a currently selected device."""
    _lazy_init()
    return torch_musa._MUSAC._musa_getDevice()


def is_available() -> bool:
    """Returns a bool indicating if MUSA is currently available."""
    return torch_musa._MUSAC._musa_getDeviceCount() > 0


def synchronize(device: _device_t = None) -> None:
    """Waits for all kernels in all streams on a MUSA device to complete."""
    _lazy_init()
    with torch_musa.device(device):
        return torch_musa._MUSAC._musa_synchronize()


@lru_cache(maxsize=1)
def device_count() -> int:
    """Returns the number of Moore Threads GPUs avaiable."""
    return torch_musa._MUSAC._musa_getDeviceCount()


class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch_musa._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch_musa._maybe_exchange_device(self.prev_idx)
        return False


def register_musa_hook():
    torch_musa._MUSAC._musa_register_device_hook()


def get_arch_list() -> List[str]:
    """Return list MUSA architectures this library was compiled for."""
    if not is_available():
        return []
    arch_flags = torch_musa._MUSAC._musa_getArchFlags()
    if arch_flags is None:
        return []
    return arch_flags.split()


def get_gencode_flags() -> str:
    r"""Return MUSA offload architecture flags this library was compiled with."""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return ""
    arch_list_ = [arch.removeprefix("mp_") for arch in arch_list]
    return " ".join([f"--offload-arch=mp_{arch}" for arch in arch_list_])


def set_sync_debug_mode(debug_mode: int | str) -> None:
    r"""Set the debug mode for musa synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations,
            if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations
        will trigger warning or error.
        In particular, operations in torch.distributed and torch.sparse namespaces
        are not covered yet.
    """
    _lazy_init()
    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )
    torch_musa._MUSAC._musa_set_sync_debug_mode(debug_mode)


def get_sync_debug_mode() -> int:
    r"""Return current value of debug mode for musa synchronizing operations."""
    _lazy_init()
    return torch_musa._MUSAC._musa_get_sync_debug_mode()


@lru_cache(maxsize=16)
def _check_bf16_tensor_supported(device: _device_t):
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        return True
    except Exception:
        return False


def is_bf16_supported(including_emulation: bool = True):
    r"""Return a bool indicating if the current MUSA device supports dtype bfloat16."""
    if not is_available():
        return False
    device = torch_musa.current_device()
    major, min = torch_musa.get_device_capability(device)

    if major * 10 + min >= 22:
        return True
    if not including_emulation:
        return False
    return _check_bf16_tensor_supported(device)


def is_tf32_supported() -> bool:
    r"""Return a bool indicating if the current MUSA device supports dtype tf32."""
    return is_bf16_supported(including_emulation=False)
