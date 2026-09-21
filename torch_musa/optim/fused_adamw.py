"""Fused Adam Optimizer"""

# pylint: disable=C0301,C0103

import torch
from typing_extensions import deprecated


@deprecated(
    "`torch.musa.optim.FusedAdamW` is deprecated. Please use `torch.optim.AdamW(fused=True)` instead.",
    category=FutureWarning,
)
def FusedAdamW(*args, **kwargs):
    return torch.optim.AdamW(*args, fused=True, **kwargs)
