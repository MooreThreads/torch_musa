# pylint: disable=missing-function-docstring, missing-module-docstring, invalid-name,redefined-outer-name, unused-import
import collections
import functools

try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
from typing import Any
import torch

__all__ = ["autocast", "custom_fwd", "custom_bwd"]


class autocast(torch.amp.autocast_mode.autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.musa.amp.autocast(args...)`` is equivalent to ``torch.autocast("musa", args...)``
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "musa"
            self.fast_dtype = dtype
            return
        super().__init__(
            "musa", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
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


# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings
# and np.ndarrays, which may be falsely detected as "Iterables."
def _cast(value, dtype):
    if isinstance(value, torch.Tensor):
        is_eligible = (
            value.is_floating_point()
            and value.device.type == "musa"
            and (value.dtype is not torch.float64)
        )
        return value.to(dtype) if is_eligible else value
    if isinstance(value, str):
        return value
    if HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    if isinstance(value, collections.abc.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    if isinstance(value, collections.abc.Iterable):
        iterable = map(lambda v: _cast(v, dtype), value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        return iterable
    return value


# custom_fwd is a decorator that may or may not be used with arguments, following
def custom_fwd(fwd=None, *, cast_inputs=None):
    if fwd is None:
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = torch.get_autocast_gpu_dtype()
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch.is_autocast_enabled()
            return fwd(*args, **kwargs)
        autocast_context = torch.is_autocast_enabled()
        args[0]._fwd_used_autocast = False
        if autocast_context:
            with autocast(enabled=False):
                return fwd(*_cast(args, cast_inputs), **_cast(kwargs, cast_inputs))
        else:
            return fwd(*args, **kwargs)

    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd):
    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(enabled=args[0]._fwd_used_autocast, dtype=args[0]._dtype):
            return bwd(*args, **kwargs)

    return decorate_bwd
