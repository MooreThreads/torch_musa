"""This package adds the memory utilities. These APIs are borrowed from cuda memory."""

# pylint: disable=unused-import, invalid-name, line-too-long, unused-argument, keyword-arg-before-vararg
import sys
import collections
import pickle
from typing import Any, Union, Tuple, Dict, Optional
import torch
from torch.types import Device
from torch._utils import _get_device_index
from torch.cuda._memory_viz import segments as _segments, memory as _memory
import torch_musa
from torch_musa.core.device import _get_musa_device_index
from ._lazy_init import _lazy_init, is_initialized
from ._utils import _get_musa_device_index


def set_per_process_memory_fraction(
    fraction, device: Union[Device, int] = None
) -> None:
    """Set memory fraction for a process.
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
        raise ValueError(f"Invalid fraction value: {fraction}. " "Allowed range: 0~1")

    torch_musa._MUSAC._musa_setMemoryFraction(fraction, device)


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
    r"""Return a dictionary of MUSA memory allocator statistics for a given device.

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
    defined size (see Memory Management section of the Cuda Semantics documentation).
    This helps avoid memory fragmentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:

    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``cudaMalloc()``.

    The caching allocator can be configured via ENV to round memory allocations in order
    to reduce fragmentation. Sometimes the overhead from rounding can be higher than
    the fragmentation it helps reduce. The following stat can be used to check if
    rounding adds too much overhead:

    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.musa.current_device`,
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
    r"""Return the result of :func:`~torch.musa.memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    device = _get_musa_device_index(device, optional=True)
    return torch_musa._MUSAC._musa_memoryStats(device)


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


def _record_memory_history(enabled="all", *args, **kwargs):
    """Enable recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currenly allocated memory.
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


def _record_memory_history_legacy(
    enabled: bool,
    record_context=True,
    trace_alloc_max_entries=1,
    trace_alloc_record_context=False,
    device: Union[Device, int] = None,
    record_context_cpp=False,
):
    torch_musa._MUSAC._musa_record_memory_history_legacy(
        enabled,
        record_context,
        trace_alloc_max_entries,
        trace_alloc_record_context,
        record_context_cpp,
    )


def _record_memory_history_impl(
    enabled: Optional[str] = "all",
    context: Optional[str] = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device: Union[Device, int] = None,
):
    torch_musa._MUSAC._musa_record_memory_history(enabled, context, stacks, max_entries)


def _snapshot(device: Union[Device, int] = None):
    return torch_musa._MUSAC._musa_memorySnapshot()


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


def _save_segment_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(_segments(snapshot))


def _save_memory_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(_memory(snapshot))


def memory_snapshot():
    """Returns a snapshot of the MUSA memory allocator state across all devices.
    Interpreting the output of this function requires familiarity with the
    memory allocator internals.
    """
    return torch_musa._MUSAC._musa_memorySnapshot()["segments"]


def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    r"""Return a human-readable printout of the current memory allocator statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
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

            # pylint: disable=consider-using-f-string
            lines.append(
                " {:<21} | {} | {} | {} | {} ".format(
                    submetric_name,
                    formatter(current, current_prefval),
                    formatter(peak, peak_prefval),
                    formatter(allocated, allocated_prefval),
                    formatter(freed, freed_prefval),
                ),
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

        # pylint: disable=consider-using-f-string
        lines.append(
            " {:<21} | {} | {} | {} | {} ".format(
                metric_name,
                formatter(current, current),
                formatter(peak, peak),
                formatter(allocated, allocated),
                formatter(freed, freed),
            ),
        )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def memory_allocated(
    device: Union[Device, int] = None, all_device: bool = False
) -> int:
    """Returns the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_musa.current_device`,
            if :attr:`device` is ``None`` (default).
        all_device (bool, optional): aggregate all devices' stats if `device` is ``None``
            and `all_device` is ``True``.

    .. note::
        This is likely less than the amount shown in `mthreads-gmi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU.
    """
    if device is not None:
        return memory_stats(device=device).get("allocated_bytes.all.current", 0)
    if all_device:
        return memory_stats_all().get("allocated_bytes.all.current", 0)
    device = torch_musa.current_device()
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(
    device: Union[Device, int] = None, all_device: bool = False
) -> int:
    """Returns the maximum GPU memory occupied by tensors in bytes for a given device.
    By default, this returns the peak allocated memory since the beginning of this program.
    :func:`~torch_musa.reset_peak_memory_stats` can be used to reset the starting point in tracking
    this metric. For example, these two functions can measure the peak allocated memory usage of
    each iteration in a training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_musa.current_device`,
            if :attr:`device` is ``None`` (default).
        all_device (bool, optional): aggregate all devices' stats if `device` is ``None``
            and `all_device` is ``True``.
    """
    if device is not None:
        return memory_stats(device=device).get("allocated_bytes.all.peak", 0)

    if all_device:
        return memory_stats_all().get("allocated_bytes.all.peak", 0)

    device = torch_musa.current_device()
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: Union[Device, int] = None, all_device: bool = False) -> int:
    """Returns the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_musa.current_device`,
            if :attr:`device` is ``None`` (default).
        all_device (bool, optional): aggregate all devices' stats if `device` is ``None``
            and `all_device` is ``True``.
    """
    if device is not None:
        return memory_stats(device=device).get("reserved_bytes.all.current", 0)

    if all_device:
        return memory_stats_all().get("reserved_bytes.all.current", 0)

    device = torch_musa.current_device()
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Union[Device, int] = None) -> int:
    """Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.musa.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.musa.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


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
    device = _get_musa_device_index(device)
    return torch.musa._MUSAC._musart.musaMemGetInfo(device)


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
    device = _get_device_index(device, optional=True)
    return torch.musa._MUSAC._musa_resetPeakMemoryStats(device)


def _set_allocator_settings(env: str):
    return torch.musa._MUSAC._musa_setMusaCachingAllocatorSettings(env)
