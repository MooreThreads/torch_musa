"""
:mod:`torch_musa.optim` is a package implementing various optimization algorithms.
"""

from .fused_lamb import FusedLAMB
from .fused_adam import FusedAdam

__all__ = ["FusedLAMB", "FusedAdam"]
