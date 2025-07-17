"""Register MUSA backend for Inductor"""

# pylint:disable=import-outside-toplevel
__all__ = ["_init_inductor_backend_registration"]

import torch
from torch._inductor import config
from .utils import _apply_util_patches
from .ir import _apply_ir_patch

# FIXME(mingyuan.wang): multi threads/processes compilation fails in both fork and spawn modes
config.compile_threads = 1


def _init_inductor_backend_registration():
    from ._aten_fallback import _make_torch_musa_aten_fallback
    from torch._inductor.codegen.common import (
        register_backend_for_device,
        register_device_op_overrides,
    )
    from .codegen.wrapper import MUSATritonWrapperCodeGen  # avoid cicular import
    from .codegen.triton import MUSATritonScheduling
    from .codegen.device_op_overrides import MUSADeviceOpOverrides
    from .runtime.triton_heuristics import _apply_triton_heuristics_patches

    # done fallback here
    _make_torch_musa_aten_fallback()

    # Register triton MUSA backend in TorchInductor
    register_device_op_overrides("musa", MUSADeviceOpOverrides())
    register_backend_for_device("musa", MUSATritonScheduling, MUSATritonWrapperCodeGen)

    # apply patches
    _apply_triton_heuristics_patches()


# `has_triton` might be called before our inductor backend registration (we use lazy
# registration to avoid cicular import), so enable util patches once torch_musa is imported
_apply_util_patches()
_apply_ir_patch()
