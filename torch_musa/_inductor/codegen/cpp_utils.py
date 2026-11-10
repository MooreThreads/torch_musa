"""CPP Utils"""

# pylint: disable=C0413, C0116
import inspect

import torch
import torch._inductor.codegen.cpp_utils

DEVICE_TO_ATEN = {"cpu": "at::kCPU", "cuda": "at::kCUDA", "musa": "at::kPrivateUse1"}

torch._inductor.codegen.cpp_utils.DEVICE_TO_ATEN = DEVICE_TO_ATEN

import torch._inductor.cpp_builder

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
        libraries += ["musa", "torch_musa"]

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

_th_cpp_torch_device_options_init = getattr(
    torch._inductor.cpp_builder.CppTorchDeviceOptions, "__init__"
)
_th_cpp_torch_device_options_signature = inspect.signature(
    _th_cpp_torch_device_options_init
)


def musa_cpp_torch_device_options_init(self, *args, **kwargs):
    _th_cpp_torch_device_options_init(self, *args, **kwargs)
    bound = _th_cpp_torch_device_options_signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    if not bound.arguments["aot_mode"]:
        return
    if bound.arguments["device_type"] != torch._C._get_privateuse1_backend_name():
        return
    # The AOTI wrapper is a standalone artifact. The installed torch headers
    # would shadow the MUSA-ported copies, and Python paths must not leak into
    # the build.
    python_include_dirs, python_lib_dirs = (
        torch._inductor.cpp_builder._get_python_related_args()
    )
    excluded_include_dirs = set(_th_include_paths("cpu", True))
    excluded_include_dirs.update(python_include_dirs)
    self._include_dirs = [
        path for path in self._include_dirs if path not in excluded_include_dirs
    ]
    self._libraries_dirs = [
        path for path in self._libraries_dirs if path not in python_lib_dirs
    ]


setattr(
    torch._inductor.cpp_builder.CppTorchDeviceOptions,
    "__init__",
    musa_cpp_torch_device_options_init,
)
