# pylint: disable=missing-function-docstring, missing-module-docstring, redefined-outer-name, unused-import, import-outside-toplevel
from importlib.util import find_spec
import torch
import torch_musa

__all__ = [
    "amp_definitely_not_available",
    "get_amp_supported_dtype",
]


def amp_definitely_not_available():
    return not (torch.musa.is_available() or find_spec("torch_musa"))


_musa_device = torch._C._get_privateuse1_backend_name()


_torch_is_autocast_enabled = torch.is_autocast_enabled


def _musa_is_autocast_enabled(*args, **kwargs) -> bool:
    if len(args) == 0 and len(kwargs) == 0:
        return _torch_is_autocast_enabled(_musa_device)
    return _torch_is_autocast_enabled(*args, **kwargs)


_torch_set_autocast_enabled = torch.set_autocast_enabled


def _musa_set_autocast_enabled(*args, **kwargs) -> None:
    if len(args) + len(kwargs) == 1:
        return _torch_set_autocast_enabled(_musa_device, *args, **kwargs)
    return _torch_set_autocast_enabled(*args, **kwargs)


def get_amp_supported_dtype():
    from torch_musa.core.device import is_bf16_supported

    choices = [torch.float16]
    if is_bf16_supported():
        choices.append(torch.bfloat16)
    return choices


def _hook_autocast_common():
    torch.is_autocast_enabled = _musa_is_autocast_enabled
    torch.set_autocast_enabled = _musa_set_autocast_enabled
