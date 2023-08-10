"""Test quantized operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.quantized as nniq
import pytest
import torch_musa

from torch_musa import testing

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
        "bias": False,
        "in_channels": 3,
        "out_channels": 32,
        "dilation": 1,
        "groups": 1,
        "relu": True,
    },
    {
        "input": torch.randn(8, 3, 32, 32, requires_grad=False),
        "accum": torch.randn(8, 16, 16, 32, requires_grad=False),
        "kernel_size": 7,
        "stride": 2,
        "padding": 3,
        "bias": False,
        "in_channels": 3,
        "out_channels": 32,
        "dilation": 1,
        "groups": 1,
        "relu": False,
    },
]
dtypes = [torch.quint8, torch.qint8]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", conv_input_data)
@pytest.mark.parametrize("dtype", dtypes)
def test_qconv2d(input_data, dtype):
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
    scale = (data.max() - data.min()) / 2**8
    zero_point = 0 - int(data.min() / scale)
    qdata = torch.quantize_per_tensor(data, scale, zero_point, torch.quint8).to("musa")
    module = torch.nn.Conv2d(**conv2d_args)
    fweight = module.weight
    fbias = module.bias
    foutput = module(data)
    if input_data["relu"]:
        foutput = torch.relu(foutput)
        qmodule = nniq.ConvReLU2d(**conv2d_args)
    else:
        qmodule = nnq.Conv2d(**conv2d_args)

    if dtype == torch.qint8:
        qweight = torch.quantize_per_tensor(
            fweight, float(fweight.abs().max() / 2**7), 0, dtype
        ).to("musa")
    else:
        qweight = torch.quantize_per_tensor(
            fweight, float(fweight.abs().max() / 2**7), 128, dtype
        ).to("musa")
    out_scale = (foutput.max() - foutput.min()) / 256
    out_zero_point = 256 - int(foutput.max() / out_scale)
    qmodule.set_weight_bias(qweight, fbias)
    qmodule.scale = out_scale
    qmodule.zero_point = out_zero_point

    qoutput = qmodule(qdata)
    assert (qoutput.dequantize().cpu() - foutput).mean() < 1e-3


# we met some numerical accuracy problem, so we deactivate qconv2d_add module UT
# will activate it when QY2 int8 conv is ready
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", conv_input_data)
@pytest.mark.parametrize("dtype", dtypes)
def test_qconv2d_add(input_data, dtype):
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
    scale = (data.max() - data.min()) / 2**8
    zero_point = 0 - int(data.min() / scale)
    qdata = torch.quantize_per_tensor(data, scale, zero_point, torch.quint8).to("musa")
    data = qdata.dequantize().cpu()

    accum = input_data["accum"]
    accum = accum.permute(0, 3, 1, 2)
    accum_scale = (accum.max() - accum.min()) / 2**8
    accum_zero_point = 0 - int(accum.min() / scale)
    qaccum = torch.quantize_per_tensor(
        accum, accum_scale, accum_zero_point, torch.quint8
    ).to("musa")
    module = torch.nn.Conv2d(**conv2d_args)
    fweight = module.weight
    fbias = module.bias
    if input_data["relu"]:
        qmodule = nniq.ConvAddReLU2d(**conv2d_args)
    else:
        qmodule = nniq.ConvAdd2d(**conv2d_args)

    if dtype == torch.qint8:
        qweight = torch.quantize_per_tensor(
            fweight, float(fweight.abs().max() / 2**7), 0, dtype
        ).to("musa")
    else:
        qweight = torch.quantize_per_tensor(
            fweight, float(fweight.abs().max() / 2**7), 128, dtype
        ).to("musa")

    module.weight = torch.nn.Parameter(qweight.dequantize().cpu())
    foutput = module(data) + accum
    if input_data["relu"]:
        foutput = torch.relu(foutput)

    out_scale = (foutput.max() - foutput.min()) / 256
    out_zero_point = 256 - int(foutput.max() / out_scale)
    qmodule.set_weight_bias(qweight, fbias)
    qmodule.scale = out_scale
    qmodule.zero_point = out_zero_point

    qoutput = qmodule(qdata, qaccum)
    assert (qoutput.dequantize().cpu() - foutput).mean() < 1e-3

    foutput = torch.quantize_per_tensor(
        foutput, float(out_scale), out_zero_point, torch.quint8
    )
    assert (
        qoutput.int_repr().cpu().to(torch.int32) - foutput.int_repr().to(torch.int32)
    ).max() <= 1


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
        fbias = fbias.to('musa')
    qmodule.set_weight_bias(qweight, fbias)
    qmodule.scale = out_scale
    qmodule.zero_point = out_zero_point

    qoutput = qmodule(qdata)
    assert (qoutput.dequantize().cpu() - foutput).mean() < 1e-3


input_datas = [
    {
        "input_a": torch.randn(128, 512),
        "input_b": torch.randn(128, 512),
        "relu": True,
    },
    {
        "input_a": torch.randn(256, 256),
        "input_b": torch.randn(256, 256),
        "relu": False,
    },
]
dtypes = [torch.quint8, torch.qint8]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_qfunctional(input_data, dtype):
    """Test quantized functional operators"""
    input_a = input_data["input_a"]
    input_b = input_data["input_b"]
    qin_a = torch.quantize_per_tensor(input_a, 0.05, 10, dtype)
    qin_b = torch.quantize_per_tensor(input_b, 0.06, 5, dtype)

    input_args = {
        "x": qin_a,
        "y": qin_b,
    }
    module = nnq.QFunctional()
    if input_data["relu"]:
        test = testing.OpTest(
            func=module.add_relu,
            input_args=input_args,
            comparators=testing.QuantizedComparator(abs_diff=1),
        )
    else:
        test = testing.OpTest(
            func=module.add,
            input_args=input_args,
            comparators=testing.QuantizedComparator(abs_diff=1),
        )
    test.check_result()
