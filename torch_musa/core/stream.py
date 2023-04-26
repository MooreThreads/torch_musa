"""Implement stream class and releated APIs"""
# pylint: disable=invalid-name, R1705, E1123, W0621, W0622, W0222, W0246
import ctypes
from typing import Any, Optional
import torch
import torch_musa
from ._utils import _dummy_type, _get_musa_device_index
from ._utils import DeviceUnion as _device_t
from .device import Device as device

if not hasattr(torch_musa._MUSAC, "_MusaStreamBase"):
    torch_musa._MUSAC.__dict__["_MusaStreamBase"] = _dummy_type("_MusaStreamBase")
    # TODO(Xiaokang Shang): Implement MUSAEvent as _MusaEventBase
    # torch_musa._MUSAC.__dict__['_MusaEventBase'] = _dummy_type('_MusaEventBase')


class Stream(torch_musa._MUSAC._MusaStreamBase):
    """Wrapper around a MUSA stream.

    A MUSA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream. Can be either
            -1 (high priority) or 0 (low priority). By default, streams have
            priority 0.

    """

    def __new__(cls, device=None, priority=0, **kwargs):
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)
        else:
            with torch_musa.device(device):
                return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    # TODO(Xiaokang Shang): event
    def wait_event(self, event):
        raise RuntimeError("Don't support event.")

    def wait_stream(self, stream):
        raise RuntimeError("Support it after event ready.")

    def query(self):
        return super().query()

    def synchronize(self):
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.musa_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.musa_stream, self.device))

    def __repr__(self):
        return (
            f"torch_musa.Stream device={self.device} musa_stream={self.musa_stream:#x}"
        )


class ExternalStream(Stream):
    r"""Wrapper around an externally allocated MUSA stream.

    This class is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This class doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this class is
       being used.

    Args:
        stream_ptr(int): Integer representation of the `musaStream_t` value.
            allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. if device is specified incorrectly,
            subsequent launches using this stream may fail.
    """

    def __new__(cls, stream_ptr, device=None, **kwargs):
        with torch_musa.device(device):
            return super(ExternalStream, cls).__new__(
                cls, stream_ptr=stream_ptr, **kwargs
            )


class StreamContext:
    r"""Context-manager that selects a given stream.

    All MUSA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        stream (stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: Optional["torch_musa.Stream"]

    def __init__(self, stream: Optional["torch_musa.Stream"]):
        self.stream = stream
        self.idx = _get_musa_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch_musa.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch_musa.default_stream(None)
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or MUSA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch_musa.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch_musa.current_stream(cur_stream.device)
        torch_musa.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no MUSA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch_musa.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch_musa.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch_musa.Stream"]) -> StreamContext:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """
    return StreamContext(stream)


def set_stream(stream: Stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch_musa._MUSAC._musa_setStream(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch_musa.current_device`, if :attr:`device` is ``None``
            (default).
    """
    streamdata = torch_musa._MUSAC._musa_getCurrentStream(
        _get_musa_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def default_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch_musa._MUSAC.current_device`, if :attr:`device` is ``None``
            (default).
    """
    streamdata = torch_musa._MUSAC._musa_getDefaultStream(
        _get_musa_device_index(device, optional=True)
    )

    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )
