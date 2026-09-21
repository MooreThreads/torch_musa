"""test skip arch check"""

# pylint: disable=C0116, C0103, W0611
import contextlib
import os
import pytest

import torch_musa
from torch_musa.core._utils import _get_musa_arch
from torch_musa.utils.musa_extension import _get_musa_arch_flags


@contextlib.contextmanager
def set_arch(key, value):
    old = os.environ.get(key, None)
    try:
        os.environ[key] = value
        yield value
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


@contextlib.contextmanager
def skip_check():
    SKIP_KEY = "TORCH_MUSA_SKIP_ARCH_CHECK"
    old = os.environ.get(SKIP_KEY, None)
    try:
        os.environ[SKIP_KEY] = "1"
        yield "1"
    finally:
        if old is None:
            os.environ.pop(SKIP_KEY, None)
        else:
            os.environ[SKIP_KEY] = old


def test_extension():
    """Test arch setting from extension"""
    arch_key = "TORCH_MUSA_ARCH_LIST"
    arch_val = "9999"
    old_val = _get_musa_arch_flags()
    with set_arch(arch_key, arch_val):
        with pytest.raises(Exception):
            _get_musa_arch_flags()
        with skip_check():
            res = _get_musa_arch_flags()
            assert res == [f"--offload-arch=mp_{arch_val}"]
    assert old_val == _get_musa_arch_flags()


def test_cure_utils():
    """Test arch setting from core utils"""
    arch_key = "TORCH_MUSA_ARCH"
    arch_val = "9999"
    old_val = int(_get_musa_arch())
    with set_arch(arch_key, arch_val):
        with pytest.raises(Exception):
            _get_musa_arch()
        with skip_check():
            res = _get_musa_arch()
            assert res == int(arch_val)
    assert old_val == _get_musa_arch()
