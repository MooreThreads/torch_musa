"""This file is about musa init."""
# pylint: disable=C0103, W0621, W0602, W0622
import traceback
import threading
from typing import Tuple, List
import torch
import torch_musa

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []


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
    return _initialized


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

        if not hasattr(torch_musa._MUSAC, "_musa_getDeviceCount"):
            raise AssertionError("Torch not compiled with MUSA enabled")

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
