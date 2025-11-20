# pylint: disable=C0301, C0114, C0103
import functools
from typing import Any

import torch

from .common import _musa_device

__all__ = ["autocast", "custom_fwd", "custom_bwd"]


class autocast(torch.amp.autocast_mode.autocast):
    r"""See :class:`torch.autocast`.

    ``torch.musa.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("musa", args...)`` instead.
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = _musa_device
            self.fast_dtype = dtype
            return
        super().__init__(
            _musa_device, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


# Preserved only for BC reasons
def _cast(value, dtype):
    return torch.amp.autocast_mode._cast(value, _musa_device, dtype)


def custom_fwd(fwd=None, *, cast_inputs=None):
    return functools.partial(torch.amp.custom_fwd, device_type=_musa_device)(
        fwd=fwd, cast_inputs=cast_inputs
    )


def custom_bwd(bwd):
    return functools.partial(torch.amp.custom_bwd, device_type=_musa_device)(bwd)
