"""Pytest hooks for operator unit tests."""

import gc
import os

import torch


_EMPTY_CACHE_AFTER_TEST = os.getenv("TORCH_MUSA_EMPTY_CACHE_AFTER_TEST") == "1"


def _clear_musa_cache():
    if not torch.musa.is_available():
        return

    gc.collect()
    torch.musa.empty_cache()


def pytest_runtest_teardown(item, nextitem):
    if not _EMPTY_CACHE_AFTER_TEST:
        return

    if nextitem is None or item.path != nextitem.path:
        _clear_musa_cache()
