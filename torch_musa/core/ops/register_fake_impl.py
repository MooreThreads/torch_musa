"""For operators that lack of FakeTensor implementation,
use torch.library.register_fake to complete registration
"""

from typing import Sequence, Tuple
import torch


# pylint: disable=unused-argument
@torch.library.register_fake("aten::_fused_rmsnorm_forward")
def fused_rmsnorm_forward(
    inp: torch.Tensor,
    normalized_shape: Sequence[int],
    eps: float,
    weight: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Register FakeTensor impl for aten::_fused_rmsnorm_forward"""
    assert isinstance(normalized_shape, (tuple, list))

    inp_shape = inp.shape
    invvar_shape = inp_shape[: -len(normalized_shape)]
    return torch.empty_like(inp), torch.empty(invvar_shape, device=inp.device)


@torch.library.register_fake("aten::_fused_rmsnorm_backward")
def fused_rmsnorm_backward(
    grad_out: torch.Tensor,
    invvar: torch.Tensor,
    inp: torch.Tensor,
    normalized_shape: Sequence[int],
    eps: float,
    weight: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Register FakeTensor impl for aten::_fused_rmsnorm_backward"""
    grad_input = torch.empty_like(inp)
    grad_weight = (
        torch.empty_like(weight)
        if weight is not None
        else torch.empty((0,), device=grad_out.device)
    )

    return grad_input, grad_weight
