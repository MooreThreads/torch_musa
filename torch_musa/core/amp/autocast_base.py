# pylint: disable=missing-function-docstring, inconsistent-return-statements, missing-class-docstring, missing-module-docstring, unused-import

import functools
import warnings

from typing import Any, Optional
import torch
from torch.types import _dtype

__all__ = ["autocast_decorator", "AutocastBase"]


def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    decorate_autocast.__script_unsupported = "@autocast() decorator is \
        not supported in script mode"  # type: ignore[attr-defined]
    return decorate_autocast


class AutocastBase:
    def __init__(
        self,
        device_type: str,
        dtype: Optional[_dtype] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = device_type
            self.fast_dtype = dtype
            # TODO: support get_autocast_gpu/cpu_dtype
            assert dtype is not None
            return
        self.device = device_type
        if self.device == "musa":
            self.fast_dtype = torch.musa.get_autocast_musa_dtype()
        elif self.device == "cpu":
            self.fast_dtype = torch.get_autocast_cpu_dtype()
        else:
            raise RuntimeError(
                "User specified autocast device_type must be 'musa' or 'cpu'"
            )
        self._cache_enabled = torch.musa.is_autocast_cache_enabled()
        if (
            enabled
            and torch.musa.amp.common.amp_definitely_not_available()
            and self.device == "musa"
        ):
            warnings.warn(
                "User provided device_type of 'musa', but MUSA is not available. Disabling"
            )
            enabled = False
        if dtype is not None:
            self.fast_dtype = dtype
        if cache_enabled is not None:
            self._cache_enabled = cache_enabled

        if self.device == "cpu":
            supported_dtype = [torch.bfloat16]
            if self.fast_dtype not in supported_dtype:
                error_message = "In CPU autocast, but the target dtype is not supported. \
                Disabling autocast.\n"
                error_message += (
                    "CPU Autocast only supports dtype of torch.bfloat16 currently."
                )
                warnings.warn(error_message)
                enabled = False
        elif self.device == "musa":
            supported_dtype = torch.musa.get_amp_supported_dtype()
            if self.fast_dtype not in supported_dtype:
                error_message = f"In {self.custom_backend_name} autocast, but the target \
                     dtype is not supported. "
                error_message += f"Disabling autocast.\n {self.custom_backend_name} \
                    Autocast only supports dtypes of "
                error_message += (
                    ", ".join(str(dtype) for dtype in supported_dtype) + " currently."
                )
                warnings.warn(error_message)
                enabled = False
        self._enabled = enabled

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            assert self.fast_dtype is not None, "autocast dtype must be set, but its None"
            return self

        self.prev_cache_enabled = torch.is_autocast_cache_enabled()
        if self.device == "cpu":
            self.prev = torch.is_autocast_cpu_enabled()
            self.prev_fastdtype = torch.get_autocast_cpu_dtype()
            torch.set_autocast_cpu_enabled(self._enabled)
            torch.set_autocast_cpu_dtype(self.fast_dtype)  # type: ignore[arg-type]
            torch.autocast_increment_nesting()
        elif self.device == "musa":
            self.prev_cache_enabled = torch.musa.is_autocast_cache_enabled()
            self.prev = torch.musa.is_autocast_musa_enabled()
            self.prev_fastdtype = torch.musa.get_autocast_musa_dtype()
            torch.musa.set_autocast_musa_enabled(self._enabled)
            torch.musa.set_autocast_musa_dtype(self.fast_dtype)
            torch.musa.autocast_increment_nesting()
        torch.musa.set_autocast_cache_enabled(self._cache_enabled)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return None
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if self.device == "cpu":
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.set_autocast_cpu_enabled(self.prev)
            torch.set_autocast_cpu_dtype(self.prev_fastdtype)
        elif self.device == "musa":
            if torch.musa.autocast_decrement_nesting() == 0:
                torch.musa.clear_autocast_cache()
            torch.musa.set_autocast_musa_enabled(self._enabled)
            torch.musa.set_autocast_musa_dtype(self.fast_dtype)
        torch.musa.set_autocast_cache_enabled(self.prev_cache_enabled)
        return False

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return autocast_decorator(self, func)
