"""
This package adds support for MUSA common functions and class.
"""

import os.path as _osp
import torch_musa

# avoid conflict with module _lazy_init
from ._lazy_init import _lazy_init as _lazy_init_musa

__all__ = [
    "cmake_prefix_path",
    "current_blas_handle",
    "current_blaslt_handle",
]
cmake_prefix_path = _osp.join(_osp.dirname(_osp.dirname(__file__)), "share", "cmake")


def current_blas_handle():
    r"""Return mublasHandle_t pointer to current muBLAS handle"""
    _lazy_init_musa()
    return torch_musa._MUSAC._musa_getCurrentBlasHandle()


def current_blaslt_handle():
    r"""Return mublasltHandle_t pointer to current muBLASLt handle"""
    _lazy_init_musa()
    return torch_musa._MUSAC._musa_getCurrentBlasLtHandle()
