"""
:mod:`torch_musa.optim` is a package implementing various optimization algorithms.
"""

from .fused_lamb import FusedLAMB

__all__ = ["FusedLAMB"]
