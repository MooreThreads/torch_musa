"""Implement Triton code generation on MUSA."""

from torch._inductor.codegen.triton import (
    TritonOverrides,
    TritonPrinter,
    TritonScheduling,
    maybe_upcast_float32,
)
from torch._inductor.virtualized import V


def _apply_triton_sqrt_patch():
    """Use MUSA libdevice sqrt while preserving upstream codegen elsewhere."""
    original_sqrt = TritonOverrides.sqrt
    original_helper_sqrt = (  # pylint: disable=protected-access
        TritonPrinter._helper_sqrt
    )

    @maybe_upcast_float32()
    def sqrt(value):
        return f"libdevice.sqrt({value})"

    def sqrt_dispatch(value):
        if V.graph.device_type == "musa":
            return sqrt(value)
        return original_sqrt(value)

    def helper_sqrt(self, expr):
        if V.graph.device_type == "musa":
            return f"libdevice.sqrt(({self._print(expr)}).to(tl.float32))"
        return original_helper_sqrt(self, expr)

    TritonOverrides.sqrt = staticmethod(sqrt_dispatch)
    TritonPrinter._helper_sqrt = helper_sqrt  # pylint: disable=protected-access


def _apply_codegen_triton_patch():
    _apply_triton_sqrt_patch()


class MUSATritonScheduling(TritonScheduling):
    pass
