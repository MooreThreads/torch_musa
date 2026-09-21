"""Fused Adam Optimizer"""

# pylint: disable=C0301,C0103

import torch
from typing_extensions import deprecated


@deprecated(
    "`torch.musa.optim.FusedAdam` is deprecated. Please use `torch.optim.Adam(fused=True)` instead.",
    category=FutureWarning,
)
def FusedAdam(*args, **kwargs):
    return torch.optim.Adam(*args, fused=True, **kwargs)
