"""Register MUSA backend for Inductor"""

# pylint:disable=import-outside-toplevel,W0718,C0103,W0612,C0413

import torch
from torch._inductor import config

# FIXME(mingyuan.wang): multi threads/processes compilation fails in both fork and spawn modes
config.compile_threads = 1

from .utils import _apply_util_patches
from .ir import _apply_ir_patch


__all__ = ["_init_inductor_backend_registration"]


def _apply_codecache_hash_patch():
    try:
        from torch._inductor import codecache as _codecache
    except (ImportError, AttributeError):
        return

    if getattr(_codecache.get_hash, "_torch_musa_patched", False):
        return

    original_get_hash = _codecache.get_hash

    def _musa_get_hash(content, extra="", hash_type="code"):
        if hash_type == "mubin":
            return _codecache.code_hash(repr(content))
        return original_get_hash(content, extra, hash_type)

    _musa_get_hash._torch_musa_patched = True
    _codecache.get_hash = _musa_get_hash


def _init_inductor_backend_registration():
    from ._aten_fallback import _make_torch_musa_aten_fallback
    from torch._inductor.codegen.common import register_backend_for_device
    from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
    from .codegen.wrapper import MUSATritonWrapperCodeGen  # avoid circular import
    from .codegen.triton import MUSATritonScheduling
    from .runtime.triton_heuristics import _apply_triton_heuristics_patches
    from .template_heuristics import _apply_template_heuristics_patches

    # done fallback here
    _make_torch_musa_aten_fallback()

    # Monkey-patch: make get_cpp_wrapper_cubin_path_name return "mubin_path" for MUSA.
    from torch._inductor import codecache as inductor_codecache
    from torch._inductor.codegen import cpp_wrapper_gpu as inductor_cpp_wrapper_gpu

    def _mubin_path_name():
        return "mubin_path"

    inductor_codecache.get_cpp_wrapper_cubin_path_name = _mubin_path_name
    inductor_cpp_wrapper_gpu.get_cpp_wrapper_cubin_path_name = _mubin_path_name

    # Register triton MUSA backend in TorchInductor
    register_backend_for_device(
        "musa", MUSATritonScheduling, MUSATritonWrapperCodeGen, CppWrapperGpu
    )

    # apply patches
    _apply_template_heuristics_patches()
    _apply_triton_heuristics_patches()


def _prepatch_triton_attrs_descriptor_for_torch25():
    import importlib

    try:
        from packaging.version import Version

        if Version(torch.__version__).release[:2] != (2, 5):
            return
    except Exception:
        return

    try:
        tbc = importlib.import_module("triton.backends.compiler")
    except Exception:
        tbc = None

    try:
        tcc = importlib.import_module("triton.compiler.compiler")
    except Exception:
        return

    from dataclasses import dataclass, is_dataclass

    if tcc is not None:
        AD_real = getattr(tcc, "AttrsDescriptor", None)
        if AD_real is None or not is_dataclass(AD_real):

            @dataclass
            class _LegacyAttrsDescriptor:
                divisible_by_16: tuple = ()
                equal_to_1: tuple = ()

            setattr(tcc, "AttrsDescriptor", _LegacyAttrsDescriptor)


# `has_triton` might be called before our inductor backend registration (we use lazy
# registration to avoid cicular import), so enable util patches once torch_musa is imported
from .codegen import device_op_overrides

_apply_codecache_hash_patch()
_apply_util_patches()
_apply_ir_patch()
_prepatch_triton_attrs_descriptor_for_torch25()
