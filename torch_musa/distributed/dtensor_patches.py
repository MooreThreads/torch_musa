"""Add some patches for DTensor here"""

from typing import Tuple, Dict

import torch
from torch.distributed.tensor import DTensor


aten = torch.ops.aten


# disable fused cross entropy loss when using tensor parallel and loss parallel,
# otherwise we need to all-gather the partial logits,
# which violates the intention of loss parallel, i.e., reduces the memory footprint
# TODO(mt-ai-infra): implement FusedCrossEntropyLossParallel, one immature idea is to be
# based on torch.autograd.Function and local_map (process DTensor)
# pylint: disable=unused-argument
def _cross_entropy_loss_2d_choice_handler(
    op_call: torch._ops.OpOverload, args: Tuple[object, ...], kwargs: Dict[str, object]
) -> object:

    return 0


_musa_customized_loss_ops = {
    aten.cross_entropy_loss_2d_choice.default: _cross_entropy_loss_2d_choice_handler
}


def _apply_dtensor_patches():
    DTensor._op_dispatcher._custom_op_handlers.update(_musa_customized_loss_ops)
