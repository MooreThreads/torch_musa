"""CPP Utils"""

# pylint: disable=C0413, C0116
import torch
import torch._inductor.codegen.cpp_utils

DEVICE_TO_ATEN = {"cpu": "at::kCPU", "cuda": "at::kCUDA", "musa": "at::kPrivateUse1"}

torch._inductor.codegen.cpp_utils.DEVICE_TO_ATEN = DEVICE_TO_ATEN

import torch._inductor.cpp_builder

_th__get_torch_related_args = getattr(
    torch._inductor.cpp_builder, "_get_torch_related_args"
)


def musa__get_torch_related_args(*args, **kwargs):
    _, libraries_dirs, libraries = _th__get_torch_related_args(*args, **kwargs)
    return [], libraries_dirs, libraries


setattr(
    torch._inductor.cpp_builder, "_get_torch_related_args", musa__get_torch_related_args
)

from torch.utils import cpp_extension
from torch_musa.utils import musa_extension

_th_include_paths = getattr(cpp_extension, "include_paths")


def musa_include_paths(device_type: str = "cpu", torch_include_dirs=True):
    if device_type == torch._C._get_privateuse1_backend_name():
        return musa_extension.include_paths(True)
    return _th_include_paths(device_type, torch_include_dirs)


setattr(cpp_extension, "include_paths", musa_include_paths)

_th_library_paths = getattr(cpp_extension, "library_paths")


def musa_library_paths(
    device_type: str = "cpu",
    torch_include_dirs: bool = True,
    cross_target_platform: str | None = None,
):
    if device_type == torch._C._get_privateuse1_backend_name():
        return musa_extension.library_paths(True)
    return _th_library_paths(device_type, torch_include_dirs, cross_target_platform)


setattr(cpp_extension, "library_paths", musa_library_paths)

_th_get_cpp_torch_device_options = getattr(
    torch._inductor.cpp_builder, "get_cpp_torch_device_options"
)


def musa_get_cpp_torch_device_options(device_type, aot_mode, compile_only):
    (
        definitions,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    ) = _th_get_cpp_torch_device_options(device_type, aot_mode, compile_only)

    if device_type == torch._C._get_privateuse1_backend_name():
        definitions.append(" USE_MUSA")
        libraries += ["musa", "musa_python"]

    return (
        definitions,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    )


setattr(
    torch._inductor.cpp_builder,
    "get_cpp_torch_device_options",
    musa_get_cpp_torch_device_options,
)
