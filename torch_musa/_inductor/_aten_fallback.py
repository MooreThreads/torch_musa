"""Fallback some operators into aten's implementation

Before falling back to aten, we should remove the operator
from the decomposition table.
"""
from typing import Sequence, Union
import torch
from torch._ops import OpOverload, OpOverloadPacket

__all__ = ["_make_torch_musa_aten_fallback"]

aten = torch.ops.aten

_decomps_to_exclude = [
    aten.nll_loss_forward,
    aten.nll_loss_backward,
    aten.embedding_dense_backward,
]

# Remove unwanted decompositions before fallback to aten

def _make_torch_musa_aten_fallback():
    # lazily import inductor modules
    # pylint: disable=import-outside-toplevel
    from torch._inductor.decomposition import decompositions, remove_decompositions

    # These ops or decomp ops work incorrect or have lower perf in triton,
    # before fixed by triton, we fallback them to aten.
    remove_decompositions(decompositions, _decomps_to_exclude)
    _fallback_to_aten(aten.nll_loss_forward)
    _fallback_to_aten(aten.nll_loss_backward)
    _fallback_to_aten(aten.embedding_dense_backward)  # torch_musa' impl has better perf


def _fallback_to_aten(
    aten_ops: Sequence[Union[OpOverload, OpOverloadPacket]],
    layout_constraint=None,
    warn=True):
    # pylint: disable=import-outside-toplevel
    from torch._inductor.lowering import make_fallback

    # Explicitly fall back some operators into aten, though inductor will
    # implicit fallback op if its not in the lowerings table.
    if not isinstance(aten_ops, (list, tuple)):
        aten_ops = [aten_ops]
    for op in aten_ops:
        if isinstance(op, OpOverloadPacket):
            for overload_name in op.overloads():
                opo = getattr(op, overload_name)
                make_fallback(opo, layout_constraint, warn)
        elif isinstance(op, OpOverload):
            make_fallback(op, layout_constraint, warn)
