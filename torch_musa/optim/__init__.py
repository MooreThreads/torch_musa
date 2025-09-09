"""
:mod:`torch_musa.optim` is a package implementing various optimization algorithms.
"""

import torch
from torch.distributed.tensor._ops.utils import register_op_strategy
from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
from torch.distributed.tensor._ops import list_pointwise_strategy

from .fused_lamb import FusedLAMB
from .fused_adam import FusedAdam
from .fused_adamw import FusedAdamW

__all__ = ["FusedLAMB", "FusedAdam", "FusedAdamW"]


# defined in fused_adam.py and fused_adamw.py
__fused_ops = [
    torch.ops.musa._fused_adam_.default,
    torch.ops.musa._fused_adam_.tensor_lr,
    torch.ops.musa._fused_adamw_.default,
    torch.ops.musa._fused_adamw_.tensor_lr,
]

# Add DTensor sharding strategies
for op in __fused_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        list_pointwise_strategy
    )
