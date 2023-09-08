"""Test quantized lineare operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.quantized as nniq
import pytest
import torch_musa

from torch_musa import testing

torch.manual_seed(41)

linear_input_data = [
    {
        "module": nnq.Linear,
        "input": torch.randn(8, 128, requires_grad=False),
        "in_features": 128,
        "out_features": 512,
        "bias": False,
        "dtype": torch.qint8,
        "relu": False,
    },
    {
        "module": nnq.Linear,
        "input": torch.randn(8, 10, 128, requires_grad=False),
        "in_features": 128,
        "out_features": 512,
        "bias": True,
        "dtype": torch.qint8,
        "relu": True,
    },
    {
        "module": nnq.Linear,
        "input": torch.randn(8, 10, 128, requires_grad=False),
        "in_features": 128,
        "out_features": 512,
        "bias": True,
        "dtype": torch.qint8,
        "relu": False,
    },
    {
        "module": nniq.LinearReLU,
        "input": torch.randn(8, 128, requires_grad=False),
        "in_features": 128,
        "out_features": 512,
        "bias": False,
        "dtype": torch.qint8,
        "relu": True,
    },
    {
        "module": nniq.LinearReLU,
        "input": torch.randn(8, 128, requires_grad=False),
        "in_features": 128,
        "out_features": 512,
        "bias": True,
        "dtype": torch.qint8,
        "relu": True,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", linear_input_data)
def test_qlinear(input_data):
    """Test quantized linear operators

    The quantized module attributes, as well as its weight and bias,
    have to be set from outside, and couldn't use module.to('musa')

    """
    data = input_data["input"]
    scale = data.abs().max() / 2**7
    qdata = torch.quantize_per_tensor(data, scale, 128, torch.quint8).to("musa")
    module = torch.nn.Linear(
        input_data["in_features"], input_data["out_features"], input_data["bias"]
    )
    fweight = module.weight
    fbias = module.bias
    foutput = module(data)
    if input_data["relu"]:
        foutput = torch.relu(foutput)

    qmodule = input_data["module"]
    qmodule = qmodule(
        input_data["in_features"],
        input_data["out_features"],
        input_data["bias"],
        input_data["dtype"],
    )
    qweight = torch.quantize_per_tensor(
        fweight, float(fweight.abs().max() / 2**7), 0, torch.qint8
    ).to("musa")
    out_scale = (foutput.max() - foutput.min()) / 256
    out_zero_point = 256 - int(foutput.max() / out_scale)
    if fbias is not None:
        fbias = fbias.to("musa")
    qmodule.set_weight_bias(qweight, fbias)
    qmodule.scale = out_scale
    qmodule.zero_point = out_zero_point

    qoutput = qmodule(qdata)
    assert (qoutput.dequantize().cpu() - foutput).mean() < 1e-3
