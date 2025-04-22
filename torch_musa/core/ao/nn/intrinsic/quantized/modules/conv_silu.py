"""intrinsic quantized conv_silu modules"""

# pylint: disable=C0103,W0622

import torch
import torch.ao.nn.intrinsic.qat
import torch.nn.functional as F
import torch.ao.nn.quantized as nnq
from torch.nn.utils import fuse_conv_bn_weights

import torch_musa


__all__ = [
    "ConvSiLU2d",
]

_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding


class ConvSiLU2d(nnq.Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and SiLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """

    _FLOAT_MODULE = torch_musa.core.ao.nn.intrinsic.ConvSiLU2d  # type: ignore[assignment]

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
        device=None,
        dtype=None,
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
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv2d_silu(
            input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvSiLU2d"

    @classmethod
    def from_float(cls, mod):
        if isinstance(mod, torch.ao.nn.intrinsic.qat.ConvBnReLU2d):
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight,
                mod.bias,
                mod.bn.running_mean,
                mod.bn.running_var,
                mod.bn.eps,
                mod.bn.weight,
                mod.bn.bias,
            )
        return super(ConvSiLU2d, cls).from_float(mod)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        assert isinstance(
            ref_qconv, torch.ao.nn.intrinsic.ConvBnReLU2d
        ), "BatchNorm2d should be fused into Conv2d before converting to reference module"
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)
