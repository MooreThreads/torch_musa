"""intrinsic qat fused modules"""

# pylint: disable=W0622,E1121,W0246

from torch import nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F

import torch_musa


__all__ = [
    "ConvBnSiLU2d",
    "ConvSiLU2d",
]


class ConvBnSiLU2d(nni.ConvBn2d):
    r"""
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and SiLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.SiLU`.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """

    # base class defines _FLOAT_MODULE as "ConvBn2d"
    _FLOAT_MODULE = torch_musa.core.ao.nn.intrinsic.ConvBnSiLU2d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = nn.Conv2d
    _FLOAT_BN_MODULE = nn.BatchNorm2d
    _FLOAT_RELU_MODULE = nn.SiLU  # type: ignore[assignment]
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = torch_musa.core.ao.nn.intrinsic.ConvSiLU2d

    def __init__(
        self,
        # Conv2d args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm2d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
        )

    def forward(self, input):
        return F.silu(nni.ConvBn2d._forward(self, input))

    @classmethod
    def from_float(cls, mod):
        return super(ConvBnSiLU2d, cls).from_float(mod)


class ConvSiLU2d(nnqat.Conv2d, nni._FusedModule):
    r"""A ConvSiLU2d module is a fused module of Conv2d and SiLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_MODULE = torch_musa.core.ao.nn.intrinsic.ConvSiLU2d
    _FLOAT_CONV_MODULE = nn.Conv2d
    _FLOAT_BN_MODULE = None
    _FLOAT_SILU_MODULE = nn.SiLU

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        return F.silu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)
        )

    @classmethod
    def from_float(cls, mod):
        return super(ConvSiLU2d, cls).from_float(mod)
