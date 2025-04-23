"""Register MUSA backend for Inductor"""

__all__ = ["_init_inductor_backend_registration"]


def _init_inductor_backend_registration():
    # lazily init inductor backend to avoid cicular import

    # pylint: disable=import-outside-toplevel
    from torch._inductor.codegen.common import register_backend_for_device
    from .codegen.wrapper import MUSATritonWrapperCodeGen  # avoid cicular import
    from .codegen.triton import MUSATritonScheduling, _apply_scheduler_patches

    from ._aten_fallback import _make_torch_musa_aten_fallback

    # done fallback here
    _make_torch_musa_aten_fallback()

    # Register triton MUSA backend in TorchInductor
    register_backend_for_device("musa", MUSATritonScheduling, MUSATritonWrapperCodeGen)

    # apply patches
    _apply_scheduler_patches()
