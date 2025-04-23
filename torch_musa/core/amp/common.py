# pylint: disable=missing-function-docstring, missing-module-docstring, redefined-outer-name, unused-import
from importlib.util import find_spec
import torch
import torch_musa
from torch_musa.core._utils import _get_musa_arch

__all__ = [
    "amp_definitely_not_available",
    "get_amp_supported_dtype",
    "is_autocast_enabled",
    "set_autocast_enabled",
    "set_autocast_dtype",
    "get_autocast_dtype",
]


def amp_definitely_not_available():
    return not (torch.musa.is_available() or find_spec("torch_musa"))


def get_amp_supported_dtype():
    return [torch.float16, torch.bfloat16, torch.float32]


def is_autocast_enabled():
    return torch_musa._MUSAC._is_autocast_musa_enabled()


def set_autocast_enabled(enable):
    return torch_musa._MUSAC._set_autocast_musa_enabled(enable)


def set_autocast_dtype(dtype):
    if dtype == torch.bfloat16:
        assert (
            _get_musa_arch() > 21
        ), 'autocast detects that the current GPU is MTT S80, please install \
        the GPU Driver and MUSA SDK for "CHUNXIAO" GPU arch. For more \
        information, please refer to \
        https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/install_guide/'
    return torch_musa._MUSAC._set_autocast_musa_dtype(dtype)


def get_autocast_dtype():
    return torch_musa._MUSAC._get_autocast_musa_dtype()
