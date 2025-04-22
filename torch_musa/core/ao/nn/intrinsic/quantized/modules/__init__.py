"""intrinsic quantized modules"""

from .conv_add import ConvAddSiLU2d
from .conv_silu import ConvSiLU2d


__all__ = [
    "ConvAddSiLU2d",
    "ConvSiLU2d",
]
