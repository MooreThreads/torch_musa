"""Test quantized operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.quantized as nniq
import pytest
import torch_musa

from torch_musa import testing

torch.manual_seed(41)

conv_input_data = [
    {
        "input": torch.randn(2, 32, 16, 16, requires_grad=False),
        "accum": torch.randn(2, 16, 16, 64, requires_grad=False),
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias": False,
        "in_channels": 32,
        "out_channels": 64,
        "dilation": 1,
        "groups": 1,
        "relu": False,
    },
    {
        "input": torch.randn(2, 32, 16, 16, requires_grad=False),
        "accum": torch.randn(2, 16, 16, 64, requires_grad=False),
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias": False,
        "in_channels": 32,
        "out_channels": 64,
        "dilation": 1,
        "groups": 1,
        "relu": True,
    },
    {
        "input": torch.randn(2, 3, 16, 16, requires_grad=False),
        "accum": torch.randn(2, 8, 8, 32, requires_grad=False),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "bias": True,
        "in_channels": 3,
        "out_channels": 32,
        "dilation": 1,
        "groups": 1,
        "relu": True,
    },
    {
        "input": torch.randn(2, 64, 16, 16, requires_grad=False),
        "accum": torch.randn(2, 8, 8, 32, requires_grad=False),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "bias": False,
        "in_channels": 64,
        "out_channels": 32,
        "dilation": 1,
        "groups": 32,
        "relu": True,
    },
    {
        "input": torch.randn(8, 3, 32, 32, requires_grad=False),
        "accum": torch.randn(8, 16, 16, 32, requires_grad=False),
        "kernel_size": 7,
        "stride": 2,
        "padding": 3,
        "bias": True,
        "in_channels": 3,
        "out_channels": 32,
        "dilation": 1,
        "groups": 1,
        "relu": False,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skip(
    # testing.get_musa_arch() < 22,  # uncomment when CI uses QY2
    reason="Quantized conv supported in QY2 or later",
)
@pytest.mark.parametrize("input_data", conv_input_data)
def test_qconv2d(input_data):
    """Test quantized conv2d operators"""
    conv2d_args = {
        "in_channels": input_data["in_channels"],
        "out_channels": input_data["out_channels"],
        "kernel_size": input_data["kernel_size"],
        "stride": input_data["stride"],
        "padding": input_data["padding"],
        "dilation": input_data["dilation"],
        "groups": input_data["groups"],
        "bias": input_data["bias"],
    }
    data = input_data["input"]
    module = torch.nn.Conv2d(**conv2d_args)
    fweight = module.weight
    fbias = module.bias
    qweight = torch.quantize_per_tensor(
        fweight, float(fweight.abs().max() / 2**7), 0, torch.qint8
    )
    module.weight = torch.nn.Parameter(qweight.dequantize())

    if input_data["relu"]:
        qmodule = nniq.ConvReLU2d(**conv2d_args)
    else:
        qmodule = nnq.Conv2d(**conv2d_args)
    qmodule.set_weight_bias(
        qweight.to("musa"), fbias.to("musa") if fbias is not None else None
    )

    if qmodule.weight().dtype == torch.quint8:
        qdata = torch.quantize_per_tensor(
            data, float(data.abs().max()) / 2**7, 128, torch.quint8
        )
        out_zero_point = 128
    else:
        qdata = torch.quantize_per_tensor(
            data, float(data.abs().max()) / 2**7, 0, torch.qint8
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
    if qmodule.weight().dtype == torch.quint8:
        foutput = torch.quantize_per_tensor(
            foutput, out_scale, out_zero_point, torch.quint8
        )
    else:
        foutput = torch.quantize_per_tensor(
            foutput, out_scale, out_zero_point, torch.qint8
        )
    assert (
        qoutput.int_repr().cpu().to(torch.int32)
        - foutput.int_repr().cpu().to(torch.int32)
    ).max() <= 1


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skip(
    # testing.get_musa_arch() < 22,  # uncomment when CI uses QY2
    reason="Quantized conv supported in QY2 or later",
)
@pytest.mark.parametrize("input_data", conv_input_data)
def test_qconv2d_add(input_data):
    """Test quantized conv2d operators"""
    conv2d_args = {
        "in_channels": input_data["in_channels"],
        "out_channels": input_data["out_channels"],
        "kernel_size": input_data["kernel_size"],
        "stride": input_data["stride"],
        "padding": input_data["padding"],
        "dilation": input_data["dilation"],
        "groups": input_data["groups"],
        "bias": input_data["bias"],
    }
    data = input_data["input"]
    accum = input_data["accum"]
    accum = accum.permute(0, 3, 1, 2)
    module = torch.nn.Conv2d(**conv2d_args)
    fweight = module.weight
    fbias = module.bias
    qweight = torch.quantize_per_tensor(
        fweight, float(fweight.abs().max() / 2**7), 0, torch.qint8
    )
    module.weight = torch.nn.Parameter(qweight.dequantize())

    if input_data["relu"]:
        qmodule = nniq.ConvAddReLU2d(**conv2d_args)
    else:
        qmodule = nniq.ConvAdd2d(**conv2d_args)
    qmodule.set_weight_bias(
        qweight.to("musa"), fbias.to("musa") if fbias is not None else None
    )

    if qmodule.weight().dtype == torch.quint8:
        qdata = torch.quantize_per_tensor(
            data, float(data.abs().max()) / 2**7, 128, torch.quint8
        )
        qaccum = torch.quantize_per_tensor(
            accum, float(accum.abs().max()) / 2**7, 128, torch.quint8
        )
        out_zero_point = 128
    else:
        qdata = torch.quantize_per_tensor(
            data, float(data.abs().max()) / 2**7, 0, torch.qint8
        )
        qaccum = torch.quantize_per_tensor(
            accum, float(accum.abs().max()) / 2**7, 0, torch.qint8
        )
        out_zero_point = 0

    foutput = module(qdata.dequantize()) + qaccum.dequantize()
    out_scale = foutput.abs().max() / 2**7
    qmodule.scale = out_scale
    qmodule.zero_point = out_zero_point

    if input_data["relu"]:
        foutput = torch.relu(foutput)
    qoutput = qmodule(qdata.to("musa"), qaccum.to("musa"))

    assert (qoutput.dequantize().cpu() - foutput).mean() < 1e-3
    if qmodule.weight().dtype == torch.quint8:
        foutput = torch.quantize_per_tensor(
            foutput, float(out_scale), out_zero_point, torch.quint8
        )
    else:
        foutput = torch.quantize_per_tensor(
            foutput, float(out_scale), out_zero_point, torch.qint8
        )
    assert (
        qoutput.int_repr().cpu().to(torch.int32) - foutput.int_repr().to(torch.int32)
    ).max() <= 1


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skip(
    # testing.get_musa_arch() < 22,  # uncomment when CI uses QY2
    reason="Quantized conv supported in QY2 or later",
)
@pytest.mark.parametrize("input_data", conv_input_data)
def test_qconv2d_silu(input_data):
    """Test quantized conv2d operators"""
    conv2d_args = {
        "in_channels": input_data["in_channels"],
        "out_channels": input_data["out_channels"],
        "kernel_size": input_data["kernel_size"],
        "stride": input_data["stride"],
        "padding": input_data["padding"],
        "dilation": input_data["dilation"],
        "groups": input_data["groups"],
        "bias": input_data["bias"],
    }
    data = input_data["input"]
    module = torch.nn.Conv2d(**conv2d_args)
    fweight = module.weight
    fbias = module.bias
    qweight = torch.quantize_per_tensor(
        fweight, float(fweight.abs().max() / 2**7), 0, torch.qint8
    )
    module.weight = torch.nn.Parameter(qweight.dequantize())

    qmodule = nniq.ConvSiLU2d(**conv2d_args)
    qmodule.set_weight_bias(
        qweight.to("musa"), fbias.to("musa") if fbias is not None else None
    )

    if qmodule.weight().dtype == torch.quint8:
        qdata = torch.quantize_per_tensor(
            data, float(data.abs().max()) / 2**7, 128, torch.quint8
        )
        out_zero_point = 128
    else:
        qdata = torch.quantize_per_tensor(
            data, float(data.abs().max()) / 2**7, 0, torch.qint8
        )
        out_zero_point = 0

    foutput = module(qdata.dequantize())
    out_scale = foutput.abs().max() / 2**7
    qmodule.scale = out_scale
    qmodule.zero_point = out_zero_point

    foutput = torch.nn.functional.silu(foutput)
    qoutput = qmodule(qdata.to("musa"))
    # assert (qoutput.dequantize().cpu() - foutput).mean() < 1e-3

    if qmodule.weight().dtype == torch.quint8:
        foutput = torch.quantize_per_tensor(
            foutput, float(out_scale), out_zero_point, torch.quint8
        )
    else:
        foutput = torch.quantize_per_tensor(
            foutput, float(out_scale), out_zero_point, torch.qint8
        )
    assert (
        qoutput.int_repr().cpu().to(torch.int32) - foutput.int_repr().to(torch.int32)
    ).max() <= 1


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skip(
    # testing.get_musa_arch() < 22,  # uncomment when CI uses QY2
    reason="Quantized conv supported in QY2 or later",
)
@pytest.mark.parametrize("input_data", conv_input_data)
def test_qconv2d_silu_add(input_data):
    """Test quantized conv2d operators"""
    conv2d_args = {
        "in_channels": input_data["in_channels"],
        "out_channels": input_data["out_channels"],
        "kernel_size": input_data["kernel_size"],
        "stride": input_data["stride"],
        "padding": input_data["padding"],
        "dilation": input_data["dilation"],
        "groups": input_data["groups"],
        "bias": input_data["bias"],
    }
    data = input_data["input"]
    accum = input_data["accum"]
    accum = accum.permute(0, 3, 1, 2)
    module = torch.nn.Conv2d(**conv2d_args)
    fweight = module.weight
    fbias = module.bias
    qweight = torch.quantize_per_tensor(
        fweight, float(fweight.abs().max() / 2**7), 0, torch.qint8
    )
    module.weight = torch.nn.Parameter(qweight.dequantize())

    qmodule = nniq.ConvAddSiLU2d(**conv2d_args)
    qmodule.set_weight_bias(
        qweight.to("musa"), fbias.to("musa") if fbias is not None else None
    )

    if qmodule.weight().dtype == torch.quint8:
        qdata = torch.quantize_per_tensor(
            data, float(data.abs().max()) / 2**7, 128, torch.quint8
        )
        qaccum = torch.quantize_per_tensor(
            accum, float(accum.abs().max()) / 2**7, 128, torch.quint8
        )
        out_zero_point = 128
    else:
        qdata = torch.quantize_per_tensor(
            data, float(data.abs().max()) / 2**7, 0, torch.qint8
        )
        qaccum = torch.quantize_per_tensor(
            accum, float(accum.abs().max()) / 2**7, 0, torch.qint8
        )
        out_zero_point = 0

    foutput = module(qdata.dequantize()) + qaccum.dequantize()
    out_scale = foutput.abs().max() / 2**7
    qmodule.scale = out_scale
    qmodule.zero_point = out_zero_point

    foutput = torch.nn.functional.silu(foutput)
    qoutput = qmodule(qdata.to("musa"), qaccum.to("musa"))

    # assert (qoutput.dequantize().cpu() - foutput).mean() < 1e-3
    if qmodule.weight().dtype == torch.quint8:
        foutput = torch.quantize_per_tensor(
            foutput, float(out_scale), out_zero_point, torch.quint8
        )
    else:
        foutput = torch.quantize_per_tensor(
            foutput, float(out_scale), out_zero_point, torch.qint8
        )
    assert (
        qoutput.int_repr().cpu().to(torch.int32) - foutput.int_repr().to(torch.int32)
    ).max() <= 1
