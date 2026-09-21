"""MUSA-specific overrides for Inductor's triton autotuning heuristics.
"""

__all__ = ["_apply_triton_heuristics_patches"]

import functools

from torch._inductor.runtime.triton_heuristics import TritonCompileResult


def _apply_scratch_defaults_patch():
    """Make the cpp wrapper pass MUSA kernels their global/profile scratch arguments.

    Triton's MUSA launcher appends two scratch pointers to every kernel's parameter list:

        MUdeviceptr global_scratch_ptr = 0;
        MUdeviceptr profile_scratch_ptr = 0;
        void *params[] = { ...args..., &global_scratch_ptr, &profile_scratch_ptr };
    """
    if getattr(TritonCompileResult.make_launcher, "_torch_musa_patched", False):
        return

    original_make_launcher = TritonCompileResult.make_launcher

    @functools.wraps(original_make_launcher)
    def make_launcher(self):
        launcher = original_make_launcher(self)
        if self.compile_meta.get("device_type") == "musa":
            if getattr(launcher, "global_scratch", None) is None:
                launcher.global_scratch = 0
            if getattr(launcher, "profile_scratch", None) is None:
                launcher.profile_scratch = 0
        return launcher

    make_launcher._torch_musa_patched = True
    TritonCompileResult.make_launcher = make_launcher


def _apply_triton_heuristics_patches():
    """Apply MUSA fixes to upstream Inductor's triton heuristics."""
    _apply_scratch_defaults_patch()
