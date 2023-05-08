"""Implement stream class and releated APIs"""
# pylint: disable=invalid-name, R1705, E1123, E1120, C0325, W0621, W0622, W0222, W0246
import ctypes
from typing import Any, Optional
import torch
import torch_musa
from ._utils import _dummy_type, _get_musa_device_index
from ._utils import DeviceUnion as _device_t
from .device import Device as device

if not hasattr(torch_musa._MUSAC, "_MusaStreamBase"):
    torch_musa._MUSAC.__dict__["_MusaStreamBase"] = _dummy_type("_MusaStreamBase")
    torch_musa._MUSAC.__dict__["_MusaEventBase"] = _dummy_type("_MusaEventBase")


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

    def wait_event(self, event):
        event.wait(self)

    def record_event(self, event=None):
        """Records an event.

        Args:
            event (torch_musa.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def wait_stream(self, stream):
        """Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.
        """
        self.wait_event(stream.record_event())

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


class Event(torch_musa._MUSAC._MusaEventBase):
    """Wrapper around a MUSA event.

    MUSA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MUSA
    streams.

    The underlying MUSA events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)
    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        return super(Event, cls).__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
        )

    @classmethod
    def from_ipc_handle(cls, device, handle):
        """Reconstruct an event from an IPC handle on the given device."""
        return super(Event, cls).from_ipc_handle(device, handle)

    def record(self, stream=None):
        """Records the event in a given stream.

        Uses ``torch_musa.current_stream()`` if no stream is specified. The
        stream's device must match the event's device."""
        if stream is None:
            stream = torch_musa.current_stream()
        super().record(stream)

    def wait(self, stream=None):
        """Makes all future work submitted to the given stream wait for this
        event.

        Use ``torch_musa.current_stream()`` if no stream is specified.

        .. note:: This is a wrapper around ``musaStreamWaitEvent()``: see
            `MUSA Event documentation`_ for more info.
        """
        if stream is None:
            stream = torch_musa.current_stream()
        super().wait(stream)

    def query(self):
        """Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event):
        """Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return super().elapsed_time(end_event)

    def synchronize(self):
        r"""Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``musaEventSynchronize()``.
        """
        super().synchronize()

    def ipc_handle(self):
        r"""Returns an IPC handle of this event. If not recorded yet, the event
        will use the current device."""
        return super().ipc_handle()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.musa_event)

    def __repr__(self):
        if self.musa_event:
            return f"<torch_musa.Event {self._as_parameter_.value:#x}>"
        else:
            return "<torch_musa.Event uninitialized>"


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
