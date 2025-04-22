"""intrinsic fused modules"""

# pylint: disable=W0221

from torch.ao.nn.intrinsic import _FusedModule
from torch.nn import Conv2d, BatchNorm2d, SiLU
from torch.nn.utils.parametrize import type_before_parametrizations


class ConvSiLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d and SiLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, silu):
        assert (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(silu) == SiLU
        ), f"Incorrect types for input modules \
            {type_before_parametrizations(conv)}{type_before_parametrizations(silu)}"
        super().__init__(conv, silu)


class ConvBnSiLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and SiLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, silu):
        assert (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(bn) == BatchNorm2d
            and type_before_parametrizations(silu) == SiLU
        ), f"Incorrect types for input modules \
            {type_before_parametrizations(conv)}{type_before_parametrizations(bn)}\
            {type_before_parametrizations(silu)}"
        super().__init__(conv, bn, silu)


class ConvAddSiLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d, add, Silu.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, add, silu):
        super().__init__(conv)
        self.add = add
        self.silu = silu

    def forward(self, x1, x2):
        return self.silu(self.add(self[0](x1), x2))
