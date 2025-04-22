"""torch_musa ao register helper"""

import torch
from .nn.intrinsic import modules as nni
from .nn.intrinsic import quantized as nniq
from .nn.intrinsic import qat


__all__ = ["_register_ao_intrinsic_modules"]


def _register_ao_intrinsic_modules():
    """register torch musa customized nni/q modules into torch.ao"""
    setattr(torch.ao.nn.intrinsic, "ConvBnSiLU2d", nni.ConvBnSiLU2d)
    setattr(torch.ao.nn.intrinsic, "ConvSiLU2d", nni.ConvSiLU2d)
    setattr(torch.ao.nn.intrinsic, "ConvAddSiLU2d", nni.ConvAddSiLU2d)
    setattr(torch.ao.nn.intrinsic.quantized, "ConvSiLU2d", nniq.ConvSiLU2d)
    setattr(torch.ao.nn.intrinsic.quantized, "ConvAddSiLU2d", nniq.ConvAddSiLU2d)
    setattr(torch.ao.nn.intrinsic.qat, "ConvBnSiLU2d", qat.ConvBnSiLU2d)
    setattr(torch.ao.nn.intrinsic.qat, "ConvSiLU2d", qat.ConvSiLU2d)
