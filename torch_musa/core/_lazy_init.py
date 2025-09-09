# pylint: disable=unused-argument
"""This file is about musa init."""

# pylint: disable=C0103, W0621, W0602, W0622, C0411, C0412, C0413, W0236, E0202, W0614, W0401, C0415
import traceback
import threading
from typing import Any, Tuple, List
import torch
import torch_musa

from ._utils import _get_musa_device_index
from .musa import *

try:
    from torch_musa._MUSAC import _musart  # type: ignore[attr-defined]
except ImportError:
    _musart = None

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []
_is_in_bad_fork = getattr(torch_musa._MUSAC, "_musa_isInBadFork", lambda: False)


class _LazySeedTracker:
    # Since seeding is memory-less, only track the latest seed.
    # Note: `manual_seed_all` followed by `manual_seed` overwrites
    # the seed on current device. We track the order of **latest**
    # calls between these two API.
    def __init__(self):
        self.manual_seed_all_cb = None
        self.manual_seed_cb = None
        self.call_order = []

    def queue_seed_all(self, cb, traceback):
        self.manual_seed_all_cb = (cb, traceback)
        # update seed_all to be latest
        self.call_order = [self.manual_seed_cb, self.manual_seed_all_cb]

    def queue_seed(self, cb, traceback):
        self.manual_seed_cb = (cb, traceback)
        # update seed to be latest
        self.call_order = [self.manual_seed_all_cb, self.manual_seed_cb]

    def get_calls(self) -> List:
        return self.call_order


_lazy_seed_tracker = _LazySeedTracker()


class DeferredMusaCallError(Exception):
    pass


default_generators: Tuple[torch._C.Generator] = ()  # type: ignore[assignment]


def _lazy_call(callable, **kwargs):
    if is_initialized():
        callable()
    else:
        # TODO(torch_deploy): this accesses linecache, which attempts to read the
        # file system to get traceback info. Patch linecache or do something
        # else here if this ends up being important.
        global _lazy_seed_tracker
        if kwargs.get("seed_all", False):
            _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
        elif kwargs.get("seed", False):
            _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
        else:
            # Don't store the actual traceback to avoid memory cycle
            _queued_calls.append((callable, traceback.format_stack()))


def is_initialized():
    """Return wherether Torch MUSA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def init():
    """Initialize PyTorch's MUSA state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for MUSA functionality will not
    be available until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's MUSA methods
    automatically initialize MUSA state on-demand.

    Does nothing if the MUSA state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    """Initialize Torch MUSA state. Will call this if `import torch_musa`.

    Does nothing if the Torch MUSA state is already initialized.
    """
    global _initialized, _queue_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        if is_initialized():
            return
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize MUSA in forked subprocess. To use MUSA with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not hasattr(torch_musa._MUSAC, "_musa_getDeviceCount"):
            raise AssertionError("Torch not compiled with MUSA enabled")
        if _musart is None:
            raise AssertionError(
                "libmusart functions unavailable. It looks like you have a broken build?"
            )

        torch_musa._MUSAC._musa_init()
        _tls.is_initializing = True
        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"MUSA call failed lazily at initialization with error: {str(e)}\n\n"
                        f"MUSA call was originally invoked at:\n\n{orig_traceback}"
                    )
                    raise DeferredMusaCallError(msg) from e

        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


def musart():
    r"""Retrieves the MUSA runtime API module.


    This function initializes the MUSA runtime environment if it is not already
    initialized and returns the MUSA runtime API module (_musart). The MUSA
    runtime API module provides access to various MUSA runtime functions.

    Args:
        ``None``

    Returns:
        module: The MUSA runtime API module (_musart).

    Raises:
        RuntimeError: If MUSA cannot be re-initialized in a forked subprocess.
        AssertionError: If PyTorch is not compiled with MUSA support or
                        if libmusart functions are unavailable.

    """
    _lazy_init()
    return _musart


class MusaError(RuntimeError):
    def __init__(self, code: int) -> None:
        msg = _musart.musaGetErrorString(_musart.musaError(code))
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    if res != _musart.musaError.success:
        raise MusaError(res)


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_musa_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch_musa._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        torch_musa._exchange_device(self.prev_idx)
        return False


@staticmethod  # type: ignore[misc]
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _MusaBase.__new__
    return super(_MusaBase, cls).__new__(cls, *args, **kwargs)


class _MusaBase:
    is_cuda = True
    is_sparse = False

    def type(self, *args, **kwargs):
        # We could use a Protocol here to tell mypy that self has `get_device` method
        # but it is only available in the typing module on Python >= 3.8
        # or on typing_extensions module on Python >= 3.6
        with device(self.get_device()):
            return super().type(*args, **kwargs)

    __new__ = _lazy_new


setattr(torch_musa, "DoubleStorage", DoubleStorage)
setattr(torch_musa, "FloatStorage", FloatStorage)
setattr(torch_musa, "LongStorage", LongStorage)
setattr(torch_musa, "IntStorage", IntStorage)
setattr(torch_musa, "ShortStorage", ShortStorage)
setattr(torch_musa, "CharStorage", CharStorage)
setattr(torch_musa, "ByteStorage", ByteStorage)
setattr(torch_musa, "HalfStorage", HalfStorage)
setattr(torch_musa, "BoolStorage", BoolStorage)
setattr(torch_musa, "BFloat16Storage", BFloat16Storage)
setattr(torch_musa, "musart", musart)
setattr(torch_musa, "MusaError", MusaError)
setattr(torch_musa, "check_error", check_error)
