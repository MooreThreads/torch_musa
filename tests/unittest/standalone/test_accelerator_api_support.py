"""Verify torch.accelerator APIs are supported by MUSA."""

# pylint: disable=C0413, C0411, C0103, W0621

import warnings

import pytest
import torch

from torch import accelerator


def _call_api(name, fn):
    """Call one accelerator API and return an error string if it is unsupported."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fn()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"{name}: {type(exc).__name__}: {exc}"
    return None


def test_torch_accelerator_apis_supported_on_musa():
    """All public torch.accelerator APIs should work on MUSA.

    The test probes every API exported by torch.accelerator.__all__.  It collects
    all unsupported APIs first, then fails once with the complete list instead of
    stopping at the first unsupported function.
    """
    assert torch.musa.is_available(), "MUSA must be available to test accelerator APIs"

    # Initialize the MUSA runtime/allocator before probing memory-stat APIs.
    tensor = torch.empty(1, device="musa")
    del tensor

    def probe_current_device_idx():
        accelerator.current_device_idx()

    def probe_set_device_index():
        prev = accelerator.current_device_index()
        try:
            accelerator.set_device_index(0)
        finally:
            accelerator.set_device_index(prev)

    def probe_set_device_idx():
        prev = accelerator.current_device_index()
        try:
            accelerator.set_device_idx(0)
        finally:
            accelerator.set_device_index(prev)

    def probe_device_index_context():
        with accelerator.device_index(0):
            pass

    def probe_set_stream():
        accelerator.set_stream(torch.musa.current_stream())

    cases = [
        ("current_accelerator", accelerator.current_accelerator),
        ("current_device_idx", probe_current_device_idx),
        ("current_device_index", accelerator.current_device_index),
        # ("get_device_capability", accelerator.get_device_capability), // unsupported on CUDA
        ("current_stream", accelerator.current_stream),
        ("device_count", accelerator.device_count),
        ("device_index", probe_device_index_context),
        ("empty_cache", accelerator.empty_cache),
        ("get_memory_info", accelerator.get_memory_info),
        ("is_available", accelerator.is_available),
        ("max_memory_allocated", accelerator.max_memory_allocated),
        ("max_memory_reserved", accelerator.max_memory_reserved),
        ("memory_allocated", accelerator.memory_allocated),
        ("memory_reserved", accelerator.memory_reserved),
        ("memory_stats", accelerator.memory_stats),
        (
            "reset_accumulated_memory_stats",
            accelerator.reset_accumulated_memory_stats,
        ),
        ("reset_peak_memory_stats", accelerator.reset_peak_memory_stats),
        ("set_device_idx", probe_set_device_idx),
        ("set_device_index", probe_set_device_index),
        ("set_stream", probe_set_stream),
        ("synchronize", accelerator.synchronize),
    ]

    unsupported = []
    for name, fn in cases:
        error = _call_api(name, fn)
        if error is not None:
            unsupported.append(error)

    if unsupported:
        pytest.fail(
            "Unsupported torch.accelerator APIs on MUSA:\n"
            + "\n".join(f" - {error}" for error in unsupported)
        )
