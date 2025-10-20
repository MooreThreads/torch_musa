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
        "input": torch.randn(8, 128, requires_grad=False),
        "in_features": 128,
        "out_features": 512,
        "bias": False,
        "dtype": torch.qint8,
        "relu": False,
    },
    {
        "input": torch.randn(8, 10, 128, requires_grad=False),
        "in_features": 128,
        "out_features": 512,
        "bias": False,
        "dtype": torch.qint8,
        "relu": False,
    },
    {
        "input": torch.randn(8, 128, requires_grad=False),
        "in_features": 128,
        "out_features": 512,
        "bias": False,
        "dtype": torch.qint8,
        "relu": True,
    },
    {
        "input": torch.randn(8, 10, 128, requires_grad=False),
        "in_features": 128,
        "out_features": 512,
        "bias": False,
        "dtype": torch.qint8,
        "relu": True,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="Quantized Lineare supported in QY2 or later"
)
@pytest.mark.parametrize("input_data", linear_input_data)
def test_qlinear(input_data):
    """Test quantized linear operators

    The quantized module attributes, as well as its weight and bias,
    have to be set from outside, and couldn't use module.to('musa')

    """
    data = input_data["input"]
    module = torch.nn.Linear(
        input_data["in_features"], input_data["out_features"], input_data["bias"]
    )
    fweight = module.weight
    fbias = module.bias
    qweight = torch.quantize_per_tensor(
        fweight, float(fweight.abs().max() / 2**7), 0, torch.qint8
    )
    module.weight = torch.nn.Parameter(qweight.dequantize())

    if input_data["relu"]:
        qmodule = nniq.LinearReLU(
            input_data["in_features"],
            input_data["out_features"],
            input_data["bias"],
            input_data["dtype"],
        )
    else:
        qmodule = nnq.Linear(
            input_data["in_features"],
            input_data["out_features"],
            input_data["bias"],
            input_data["dtype"],
        )
    qmodule.set_weight_bias(
        qweight.to("musa"), fbias.to("musa") if fbias is not None else None
    )

    qdata = torch.quantize_per_tensor(
        data, float(data.abs().max() / 2**7), 0, torch.qint8
    )
    out_zero_point = 0

    foutput = module(qdata.dequantize())

    out_scale = float(foutput.abs().max() / 2**7)
    qmodule.scale = out_scale
    qmodule.zero_point = out_zero_point
    qoutput = qmodule(qdata.to("musa"))

    if input_data["relu"]:
        foutput = torch.relu(foutput)

    assert (qoutput.dequantize().cpu() - foutput).mean() < 1e-3
    foutput = torch.quantize_per_tensor(foutput, out_scale, out_zero_point, torch.qint8)
    assert (
        qoutput.int_repr().cpu().to(torch.int32)
        - foutput.int_repr().cpu().to(torch.int32)
    ).max() <= 1
