"""This package adds the memory utilities. These APIs are borrowed from cuda memory."""

# pylint: disable=C0301, W1113, W0246, C0415, unused-import, invalid-name, too-many-statements, too-many-locals, unused-argument, unspecified-encoding
import sys
import ctypes
import contextlib
import collections
import pickle
import warnings
from inspect import signature
from typing import Any, Union, Tuple, Dict, Optional
from typing_extensions import deprecated
import torch
from torch.types import Device
from torch.cuda._memory_viz import segments as _segments, memory as _memory
import torch_musa
import torch_musa._MUSAC
from torch_musa.core.device import _get_musa_device_index


__all__ = [
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_enable",
    "get_per_process_memory_fraction",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "reset_accumulated_host_memory_stats",
    "reset_peak_host_memory_stats",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",
    "list_gpu_processes",
    "mem_get_info",
    "get_allocator_backend",
    "MUSAPluggableAllocator",
    "change_current_allocator",
    "MemPool",
    "MemPoolContext",
    "use_mem_pool",
    # remove visibility to torch.musa in the future
    # "memory_stats_all",
    # "_record_memory_history",
    # "_dump_snapshot",
    # "_set_allocator_settings",
]


from torch_musa._MUSAC import (
    _musa_MUSAAllocator,
    _musa_beginAllocateCurrentThreadToPool,
    _musa_beginAllocateToPool,
    _musa_endAllocateToPool,
    _MemPool,
    _MemPoolContext,
    _musa_releasePool,
)

from ._lazy_init import _lazy_init, is_initialized
from ._utils import _get_musa_device_index


def caching_allocator_alloc(size, device: Union[Device, int] = None, stream=None):
    r"""Perform a memory allocation using the MUSA memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch_musa.caching_allocator_delete`.

    Args:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MUSA device is used.
        stream (torch_musa.Stream or int, optional): selected stream. If is
            ``None`` then the default stream for the selected device is used.

    .. note::
        See :ref:`musa-memory-management` for more details about GPU memory
        management.
    """
    if device is None:
        device = torch_musa.current_device()
    device = _get_musa_device_index(device)
    if stream is None:
        stream = torch_musa.current_stream(device)
    if isinstance(stream, torch_musa.Stream):
        stream = stream.musa_stream
    if not isinstance(stream, int):
        raise TypeError(
            "Invalid type for stream argument, must be "
            "`torch_musa.Stream` or `int` representing a pointer "
            "to a existing stream"
        )
    with torch_musa.device(device):
        return torch_musa._MUSAC._musa_musaCachingAllocator_raw_alloc(size, stream)


def caching_allocator_delete(mem_ptr):
    r"""Delete memory allocated using the MUSA memory allocator.

    Memory allocated with :func:`~torch_musa.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Args:
        mem_ptr (int): memory address to be freed by the allocator.

    .. note::
        See :ref:`musa-memory-management` for more details about GPU memory
        management.
    """
    torch_musa._MUSAC._musa_musaCachingAllocator_raw_delete(mem_ptr)


def caching_allocator_enable(value: bool = True) -> None:
    r"""Enable or disable the MUSA memory allocator. On by default."""
    if is_initialized():
        torch_musa._MUSAC._musa_musaCachingAllocator_enable(value)


def set_per_process_memory_fraction(
    fraction, device: Union[Device, int] = None
) -> None:
    r"""Set memory fraction for a process.
    The fraction is used to limit an caching allocator to allocated memory on a MUSA device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MUSA device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
    _lazy_init()
    if device is None:
        device = torch_musa.current_device()
    device = _get_musa_device_index(device)
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 1:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~1")

    torch_musa._MUSAC._musa_setMemoryFraction(fraction, device)


def get_per_process_memory_fraction(device: Union[Device, int] = None) -> float:
    r"""Get memory fraction for a process.

    Args:
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MUSA device is used.

    Returns:
        memory fraction, in range 0~1. Allowed memory equals total_memory * fraction.
    """
    _lazy_init()
    if device is None:
        device = torch_musa.current_device()
    device = _get_musa_device_index(device)
    return torch_musa._MUSAC._musa_getMemoryFraction(device)


def empty_cache():
    """Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other musa application and visible in
    `mthreads-gmi`.

    .. note::
        :func:`~torch_musa.empty_cache` doesn't increase the amount of musa
        memory available for PyTorch. However, it may help reduce fragmentation
        of GPU memory in certain cases.
    """
    if is_initialized():
        torch_musa._MUSAC._musa_emptyCache()


def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    """Returns a dictionary of NPU memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:
    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``musaMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:
    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).

    Metric type:
    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:
    - ``"num_alloc_retries"``: number of failed ``musaMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.

    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the MUSA Semantics documentation).
    This helps avoid memory framentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:
    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``musaMalloc()``.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch_musa.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def memory_stats_as_nested_dict(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Return the result of :func:`~torch_musa.memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    device = _get_musa_device_index(device, optional=True)
    return torch_musa._MUSAC._musa_memoryStats(device)


def reset_accumulated_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the MUSA memory allocator.

    See :func:`~torch.musa.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.musa.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`musa-memory-management` for more details about GPU memory
        management.
    """
    device = _get_musa_device_index(device, optional=True)
    return torch_musa._MUSAC._musa_resetAccumulatedMemoryStats(device)


def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Resets the "peak" stats tracked by the MUSA memory allocator.

    See :func:`~torch.musa.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.musa.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`musa-memory-management` for more details about GPU memory
        management.
    """
    device = _get_musa_device_index(device, optional=True)
    return torch.musa._MUSAC._musa_resetPeakMemoryStats(device)


def host_memory_stats() -> Dict[str, Any]:
    r"""Return a dictionary of MUSA host memory allocator statistics.

     The return value of this function is a dictionary of statistics, each of
     which is a non-negative integer.

     Core statistics:

     - ``"allocated.{current,peak,allocated,freed}"``:
       number of allocation requests received by the memory allocator.
     - ``"allocated_bytes.{current,peak,allocated,freed}"``:
       amount of allocated memory.
     - ``"segment.{current,peak,allocated,freed}"``:
       number of reserved segments from ``musaHostAlloc()``.
     - ``"reserved_bytes.{current,peak,allocated,freed}"``:
       amount of reserved memory.

     For these core statistics, values are broken down as follows.

     Metric type:

     - ``current``: current value of this metric.
     - ``peak``: maximum value of this metric.
     - ``allocated``: historical total increase in this metric.
     - ``freed``: historical total decrease in this metric.

     In addition to the core statistics, we also provide some simple event
     counters:

     - ``"num_host_alloc"``: number of MUSA host allocation calls. This includes
       both musaHostAlloc and musaHostRegister.
     - ``"num_host_free"``: number of MUSA host free calls. This includes both
       musaHostFree and musaHostUnregister.

     Finally, we also provide some simple timing counters:

     - ``"host_alloc_time.{total,max,min,count,avg}"``:
       timing of allocation requests going through MUSA calls.
     - ``"host_free_time.{total,max,min,count,avg}"``:
       timing of free requests going through MUSA calls.

    For these timing statistics, values are broken down as follows.

     Metric type:

     - ``total``: total time spent.
     - ``max``: maximum value per call.
     - ``min``: minimum value per call.
     - ``count``: number of times it was called.
     - ``avg``: average time per call.
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = host_memory_stats_as_nested_dict()
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def host_memory_stats_as_nested_dict() -> Dict[str, Any]:
    r"""Return the result of :func:`~torch_musa.host_memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    return torch_musa._MUSAC._musa_hostMemoryStats()


def reset_accumulated_host_memory_stats() -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the host memory allocator.

    See :func:`~torch_musa.host_memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict.
    """
    return torch_musa._MUSAC._musa_resetAccumulatedHostMemoryStats()


def reset_peak_host_memory_stats() -> None:
    r"""Reset the "peak" stats tracked by the host memory allocator.

    See :func:`~torch_musa.host_memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.
    """
    return torch_musa._MUSAC._musa_resetPeakHostMemoryStats()


def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device.

    See :func:`~torch.musa.max_memory_allocated` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.musa.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.musa.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`musa-memory-management` for more details about GPU memory
        management.
    """
    warnings.warn(
        "torch.musa.reset_max_memory_allocated now calls torch.musa.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def reset_max_memory_cached(device: Union[Device, int] = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.

    See :func:`~torch.musa.max_memory_cached` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.musa.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.musa.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`musa-memory-management` for more details about GPU memory
        management.
    """
    warnings.warn(
        "torch.musa.reset_max_memory_cached now calls torch.musa.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def memory_stats_all():
    ret_dict = collections.defaultdict(int)
    num_device = torch_musa.device_count()
    for i in range(num_device):
        for k, v in memory_stats(i).items():
            if k.split(".")[-1] == "peak":
                ret_dict[k] = max(ret_dict[k], v)
            else:
                ret_dict[k] += v
    return ret_dict


def _record_memory_history_legacy(
    enabled: bool,
    record_context=True,
    trace_alloc_max_entries=1,
    trace_alloc_record_context=False,
    device: Union[Device, int] = None,
    record_context_cpp=False,
    clear_history=False,
    compile_context=False,
    global_record_annotations=False,
):
    torch_musa._MUSAC._musa_record_memory_history_legacy(
        enabled,
        record_context,
        trace_alloc_max_entries,
        trace_alloc_record_context,
        record_context_cpp,
        clear_history,
        compile_context,
        global_record_annotations,
    )


def _record_memory_history_impl(
    enabled: Optional[str] = "all",
    context: Optional[str] = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device: Union[Device, int] = None,
    clear_history: bool = False,
    compile_context: bool = False,
    global_record_annotations: bool = False,
):
    torch_musa._MUSAC._musa_record_memory_history(
        enabled,
        context,
        stacks,
        max_entries,
        clear_history,
        compile_context,
        global_record_annotations,
    )


def _record_memory_history(enabled="all", *args, **kwargs):
    """Enable recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in
    :func:`torch_musa.memory._snapshot()`.

    In addition to keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`torch_musa.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    Buffer behavior
    ---------------

    This will store up to `max_entries` instances of `TraceEntry` when enabled.
    Python trace collection defaults to `sys.maxsize`, meaning long-running
    or indefinitely running jobs should set a reasonable limit to avoid excessive
    memory use. Expect each entry to be several KB.

    Longer running workflows or those with smaller `max_entries` values will only
    store the last accumulated `max_entries` entries, meaning new entries overwrite
    older entries.

    C++ implementation for reference to ring buffer implementation:

    .. code-block:: cpp

        if (record_history) {
          if (alloc_trace->size() < alloc_trace_max_entries_) {
            alloc_trace->emplace_back(te);
          } else {
            (*alloc_trace)[alloc_trace_next++] = te;
            if (alloc_trace_next == alloc_trace_max_entries_) {
              alloc_trace_next = 0;
            }
          }
        }

    Latency impact
    --------------

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currently allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal[None, "state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks
            `"all"`, additionally include C++ frames
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
    """
    if isinstance(enabled, bool):
        return _record_memory_history_legacy(enabled, *args, **kwargs)
    return _record_memory_history_impl(enabled, *args, **kwargs)


_record_memory_history.__signature__ = signature(_record_memory_history_impl)  # type: ignore[attr-defined]


def _snapshot(device: Union[Device, int] = None):
    """Save a snapshot of MUSA memory state at the time it was called.

    The state is represented as a dictionary with the following structure.

    .. code-block:: python

        class Snapshot(TypedDict):
            segments: List[Segment]
            device_traces: List[List[TraceEntry]]


        class Segment(TypedDict):
            # Segments are memory returned from a musaMalloc call.
            # The size of reserved memory is the sum of all Segments.
            # Segments are cached and reused for future allocations.
            # If the reuse is smaller than the segment, the segment
            # is split into more then one Block.
            # empty_cache() frees Segments that are entirely inactive.
            address: int
            total_size: int  #  musaMalloc'd size of segment
            stream: int
            segment_type: Literal["small", "large"]  # 'large' (>1MB)
            allocated_size: int  # size of memory in use
            active_size: int  # size of memory in use or in active_awaiting_free state
            blocks: List[Block]


        class Block(TypedDict):
            # A piece of memory returned from the allocator, or
            # current cached but inactive.
            size: int
            requested_size: int  # size requested during malloc, may be smaller than
            # size due to rounding
            address: int
            state: Literal[
                "active_allocated",  # used by a tensor
                "active_awaiting_free",  # waiting for another stream to finish using
                # this, then it will become free
                "inactive",
            ]  # free for reuse
            frames: List[Frame]  # stack trace from where the allocation occurred


        class Frame(TypedDict):
            filename: str
            line: int
            name: str


        class TraceEntry(TypedDict):
            # When `torch.musa.memory._record_memory_history()` is enabled,
            # the snapshot will contain TraceEntry objects that record each
            # action the allocator took.
            action: Literal[
                "alloc"  # memory allocated
                "free_requested",  # the allocated received a call to free memory
                "free_completed",  # the memory that was requested to be freed is now
                # able to be used in future allocation calls
                "segment_alloc",  # the caching allocator ask musaMalloc for more memory
                # and added it as a segment in its cache
                "segment_free",  # the caching allocator called musaFree to return memory
                # to musa possibly trying free up memory to
                # allocate more segments or because empty_caches was called
                "oom",  # the allocator threw an OOM exception. 'size' is
                # the requested number of bytes that did not succeed
                "snapshot",  # the allocator generated a memory snapshot
                # useful to coorelate a previously taken
                # snapshot with this trace
            ]
            addr: int  # not present for OOM
            frames: List[Frame]
            size: int
            stream: int
            device_free: int  # only present for OOM, the amount of
            # memory musa still reports to be free

    Returns:
        The Snapshot dictionary object
    """
    return torch_musa._MUSAC._musa_memorySnapshot(None)


def _dump_snapshot(filename="dump_snapshot.pickle"):
    """
    Save a pickled version of the `torch.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Args:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
    """
    s = _snapshot()
    with open(filename, "wb") as f:
        pickle.dump(s, f)


def _select_format_flamegraph(flamegraph_lines):
    from os import getenv  # pylint: disable=C0415
    from torch.cuda._memory_viz import (
        format_flamegraph as _format_flamegraph,
    )  # pylint: disable=C0415

    local_script = getenv("FLAMEGRAPH_PL_SCRIPT", None)
    return _format_flamegraph(flamegraph_lines, local_script)


def _save_segment_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()

    with open(filename, "w") as f:
        f.write(_segments(snapshot, format_flamegraph=_select_format_flamegraph))


def _save_memory_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with open(filename, "w") as f:
        f.write(_memory(snapshot, format_flamegraph=_select_format_flamegraph))


def memory_snapshot(mempool_id=None):
    r"""Return a snapshot of the MUSA memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    .. note::
        See :ref:`musa-memory-management` for more details about GPU memory
        management.
    """
    return torch_musa._MUSAC._musa_memorySnapshot(mempool_id)["segments"]


def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    """Returns a human-readable printout of the current memory allocator statistics for a given
    device.

    This can be useful to display periodically during training, or when handling out-of-memory
    exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch_musa.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary (default: False).
    """
    device = _get_musa_device_index(device, optional=True)
    stats = memory_stats(device=device)

    def _format_size(sz, pref_sz):
        prefixes = ["B  ", "KiB", "MiB", "GiB", "TiB", "PiB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return f"{cnt:7d} {prefix} "

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("requested_bytes", "Requested memory", _format_size),
        ("reserved_bytes", "GPU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "GPU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} Torch MUSA memory summary, device ID {device:<17d} ")
    lines.append("-" * 75)
    lines.append(
        "  {_:9} MUSA OOMs: {num_ooms:<12d} | {_:6} musaMalloc retries: {num_alloc_retries:<8d}  "
    )
    lines.append("=" * 75)
    lines.append(
        "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "
    )

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = (
            None,
            None,
            None,
            None,
        )

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(
                f" {submetric_name:<21} | {formatter(current, current_prefval)} | {formatter(peak, peak_prefval)} | "
                f"{formatter(allocated, allocated_prefval)} | {formatter(freed, freed_prefval)} ",
            )

    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize GPU segments", _format_count),
    ]

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)

        prefix = metric_key + "."

        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]

        lines.append(
            f" {metric_name:<21} | {formatter(current, current)} | {formatter(peak, peak)} | "
            f"{formatter(allocated, allocated)} | {formatter(freed, freed)} ",
        )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def memory_allocated(device: Union[Device, int] = None) -> int:
    """Returns the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_musa.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `mthreads-gmi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU.
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: Union[Device, int] = None) -> int:
    """Returns the maximum GPU memory occupied by tensors in bytes for a given device.
    By default, this returns the peak allocated memory since the beginning of this program.
    :func:`~torch_musa.reset_peak_memory_stats` can be used to reset the starting point in tracking
    this metric. For example, these two functions can measure the peak allocated memory usage of
    each iteration in a training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_musa.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: Union[Device, int] = None) -> int:
    """Returns the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_musa.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Union[Device, int] = None) -> int:
    """Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.
    By default, this returns the peak cached memory since the beginning of this program.
    :func:`~torch_musa.reset_peak_memory_stats` can be used to reset the starting point in tracking
    this metric. For example, these two functions can measure the peak cached memory amount of each
    iteration in a training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_musa.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


@deprecated(
    "`torch.musa.memory_cached` has been renamed to `torch.musa.memory_reserved`",
    category=FutureWarning,
)
def memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.musa.memory_reserved`."""
    return memory_reserved(device=device)


@deprecated(
    "`torch.musa.max_memory_cached` has been renamed to `torch.musa.max_memory_reserved`",
    category=FutureWarning,
)
def max_memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.musa.max_memory_reserved`."""
    return max_memory_reserved(device=device)


def list_gpu_processes(device: Union[Device, int] = None) -> str:
    r"""Return a human-readable printout of the running processes and their GPU memory use for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.musa.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    import shutil
    import subprocess

    if device is None:
        device = torch.musa.current_device()
    device = _get_musa_device_index(device, optional=True)

    mthreads_gmi = shutil.which("mthreads-gmi")
    if mthreads_gmi is None:
        return "mthreads-gmi not found, please install mthreads-gmi"

    try:
        return subprocess.check_output(
            [mthreads_gmi, "--id", str(device)], stderr=subprocess.STDOUT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            return subprocess.check_output(
                [mthreads_gmi], stderr=subprocess.STDOUT, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "musa driver can't be loaded, is musa enabled?"


def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]:
    r"""Returns the global free and total GPU memory occupied for a given
    device using musaMemGetInfo.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.musa.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`musa-memory-management` for more
        details about GPU memory management.
    """
    if device is None:
        device = torch.musa.current_device()
    device = _get_musa_device_index(device, optional=True)
    return torch.musa.musart().musaMemGetInfo(device)


def _set_allocator_settings(env: str):
    return torch.musa._MUSAC._musa_musaCachingAllocator_set_allocator_settings(env)


def get_allocator_backend() -> str:
    """Return a string describing the active allocator backend"""
    return torch_musa._MUSAC._musa_getAllocatorBackend()


class _MUSAAllocator:
    """Wrapper over internal MUSA memory allocators."""

    def __init__(self, allocator: _musa_MUSAAllocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class MUSAPluggableAllocator(_MUSAAllocator):
    """MUSA pluggable allocator, which loaded from a so file."""

    # pylint: disable=super-init-not-called
    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        """Memory allocators are compiled in .so files and loaded dynamically using ctypes."""
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch_musa._MUSAC._musa_customAllocator(alloc_fn, free_fn)


def change_current_allocator(allocator: _MUSAAllocator) -> None:
    """Change the currently used memory allocator to be the one provided.

    If the current allocator has already been used/initialized, this function will error.
    """
    torch_musa._MUSAC._musa_changeCurrentAllocator(allocator.allocator())


def _get_current_allocator() -> _MUSAAllocator:
    """Return the allocator being currently used."""
    return _MUSAAllocator(torch_musa._MUSAC._musa_getAllocator())


class MemPool(_MemPool):
    """MemPool represents a pool of memory in a caching allocator. Currently,
    it's just the ID of the pool object maintained in the MUSACachingAllocator.

    Args:
        allocator(_musa_MUSAAllocator, optional): a _musa_MUSAAllocator object
            that can be used to define how memory gets allocated in the pool. If
            :attr:`allocator` is ``None`` (default), memory allocation follows
            the default/current configuration of the MUSACachingAllocator.
        use_on_oom(bool): a bool that indicates if this pool can be used as a
            last resort if a memory allocation outside of the pool fails due to
            Out Of Memory. This is False by default.

    """

    def __init__(
        self,
        allocator: Optional[_musa_MUSAAllocator] = None,
        use_on_oom: bool = False,
    ):
        super().__init__(allocator, True, use_on_oom)

    @property
    def id(self) -> Tuple[int, int]:
        """Returns the ID of this pool as a tuple of two ints."""
        return super().id

    @property
    def allocator(self) -> Optional[_musa_MUSAAllocator]:
        """Returns the allocator this MemPool routes allocations to."""
        return super().allocator

    def use_count(self) -> int:
        r"""Returns the reference count of this pool."""
        return super().use_count()

    def snapshot(self):
        r"""Return a snapshot of the MUSA memory allocator pool state across all
        devices.

        Interpreting the output of this function requires familiarity with the
        memory allocator internals.

        .. note::
            See :ref:`musa-memory-management` for more details about GPU memory
            management.
        """
        snapshot = torch.musa.memory_snapshot(self.id)
        return snapshot


class MemPoolContext(_MemPoolContext):
    """MemPoolContext holds the currently active pool and stashed the previous
    pool. On deletion it makes the previous pool active.
    """

    def __init__(self, pool: MemPool):
        if hasattr(_MemPoolContext, "__init__"):
            super().__init__(pool)

    @staticmethod
    def active_pool() -> Optional[_MemPool]:
        """Returns the active MemPool"""
        return _MemPoolContext.activate_pool()


@contextlib.contextmanager
def use_mem_pool(pool: MemPool, device: Union[Device, int] = None):
    r"""A context manager that routes allocations to a given pool.

    Args:
        pool(torch.musa.MemPool): a MemPool object to be made active so that
            allocations route to this pool.
        device (torch.device or int, optional): selected device. Uses MemPool on
            the current device, given by :func:`~torch.musa.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This context manager makes only current thread's allocations route to
        the given pool. If a new thread is spawned inside the context manager
        (e.g. by calling backward) the allocations in that thread will not
        route to the given pool.
    """
    device_idx = (
        torch.musa.current_device()
        if device is None
        else _get_musa_device_index(device)
    )
    _musa_beginAllocateCurrentThreadToPool(device_idx, pool.id)
    try:
        yield
    finally:
        _musa_endAllocateToPool(device_idx, pool.id)
        _musa_releasePool(device_idx, pool.id)
