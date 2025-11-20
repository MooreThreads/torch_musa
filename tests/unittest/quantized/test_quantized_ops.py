"""Test quantized operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, invalid-name, unexpected-keyword-arg
import pytest
import torch
import torch.ao.nn.quantized as nnq
import torch_musa

from torch_musa import testing

torch.manual_seed(41)


def function(input_data, func, dtype=None):
    if dtype is not None and isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(
        func=func, input_args=input_data, comparators=testing.QuantizedComparator()
    )
    test.check_result()


dtype = [torch.quint8, torch.qint8]
reduce_range = [True, False]
input_data_per_tensor = [
    {
        "input": torch.randn(2, 3, 4),
    },
    {
        "input": torch.randn(2, 3, 4, 5),
    },
    {
        "input": torch.randn(2, 16, 32, 32),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data_per_tensor)
@pytest.mark.parametrize("dtype", dtype)
def test_quantize_per_tensor(input_data, dtype):
    inputs = {
        "input": input_data["input"],
        "scale": input_data["input"].abs().max() / 2**7,
        "zero_point": 0 if dtype == torch.qint8 else 128,
        "dtype": dtype,
    }
    test = testing.OpTest(
        func=torch.quantize_per_tensor,
        input_args=inputs,
        comparators=testing.QuantizedComparator(),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data_per_tensor)
@pytest.mark.parametrize("reduce_range", reduce_range)
def test_quantize_per_tensor_dynamic(input_data, reduce_range):
    test = testing.OpTest(
        func=torch.quantize_per_tensor_dynamic,
        input_args={
            "input": input_data["input"],
            "dtype": torch.quint8,
            "reduce_range": reduce_range,
        },
        comparators=testing.QuantizedComparator(),
    )
    test.check_result()


input_data_per_channel = [
    {
        "input": torch.randn(2, 3),
        "scales": torch.Tensor([0.1, 0.01]),
        "zero_points": torch.Tensor([10, 1]),
        "axis": 0,
    },
    {
        "input": torch.randn(2, 3, 4, 5, 6),
        "scales": torch.Tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
        "zero_points": torch.Tensor([1, 1, 1, 1, 1]),
        "axis": 3,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data_per_channel)
@pytest.mark.parametrize("dtype", dtype)
def test_quantize_per_channel(input_data, dtype):
    if dtype is not torch.qint8 and isinstance(input_data["input"], torch.Tensor):
        input_data["dtype"] = dtype
        test = testing.OpTest(
            func=torch.quantize_per_channel,
            input_args=input_data,
            comparators=testing.QuantizedComparator(is_per_tensor=False),
        )
        test.check_result()


input_data_int_per_tensor = [
    {"input": torch.randint(0, 100, size=(2, 3, 4)), "scale": 0.1, "zero_point": 1},
    {
        "input": torch.randint(0, 100, size=(2, 3, 4, 5)),
        "scale": 0.0143444,
        "zero_point": 127,
    },
]
int_dtype = [torch.uint8, torch.int32, torch.int8]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data_int_per_tensor)
@pytest.mark.parametrize("dtype", int_dtype)
def test_make_per_tensor_quantized_tensor(input_data, dtype):
    function(input_data, torch._make_per_tensor_quantized_tensor, dtype)


input_data_int_per_channel = [
    {
        "input": torch.randint(0, 100, size=(2, 3, 4)),
        "scale": torch.Tensor([0.1, 0.01]),
        "zero_point": torch.Tensor([1, 1]),
        "axis": 0,
    },
    {
        "input": torch.randint(0, 100, size=(2, 3, 4, 5)),
        "scale": torch.Tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
        "zero_point": torch.Tensor([0, 1, 2, 3, 4]),
        "axis": 3,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data_int_per_channel)
@pytest.mark.parametrize("dtype", int_dtype)
def test_make_per_channel_quantized_tensor(input_data, dtype):
    if dtype is not torch.int32 and isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
        test = testing.OpTest(
            func=torch._make_per_channel_quantized_tensor,
            input_args=input_data,
            comparators=testing.QuantizedComparator(is_per_tensor=False),
        )
        test.check_result()


input_as_stride = [
    {
        "input": torch.quantize_per_tensor(torch.randn(3, 3), 0.1, 127, torch.quint8),
        "size": (2, 2),
        "stride": (1, 2),
    },
    {
        "input": torch.quantize_per_tensor(torch.randn(4, 6), 0.1, 1, torch.qint8),
        "size": (2, 2),
        "stride": (1, 2),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_as_stride)
def test_as_strided(input_data):
    function(input_data, torch.as_strided)


input_fill = [
    {
        "input": torch.quantize_per_tensor(torch.randn(3, 4), 0.1, 127, torch.quint8),
        "value": 0.1,
    },
    {
        "input": torch.quantize_per_tensor(torch.randn(3, 4, 6), 0.1, 1, torch.qint8),
        "value": torch.tensor(1.1),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_fill)
def test_fill_(input_data):
    function(input_data, torch.fill_)


input_tensor_shape = [
    {
        "input": torch.quantize_per_tensor(
            torch.randn(3, 1, 4), 0.1, 127, torch.quint8
        ),
        "dim": 1,
    },
    {
        "input": torch.quantize_per_tensor(torch.randn(3, 1, 4), 0.1, 0, torch.qint8),
        "dim": 1,
    },
    {
        "input": torch.quantize_per_tensor(
            torch.randn(3, 4, 6, 1), 0.1, 1, torch.qint8
        ),
        "dim": 3,
    },
    {
        "input": torch.quantize_per_tensor(
            torch.randn(3, 4, 6, 1), 0.1, 1, torch.qint8
        ),
        "dim": 1,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_tensor_shape)
def test_squeeze(input_data):
    function(input_data, torch.squeeze)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_tensor_shape)
def test_unsqueeze(input_data):
    function(input_data, torch.unsqueeze)


input_index_select = [
    {
        "input": torch.quantize_per_tensor(
            torch.randn(3, 4, 6), 0.1, 127, torch.quint8
        ),
        "dim": 0,
        "index": torch.tensor([0, 2], dtype=torch.int32),
    },
    {
        "input": torch.quantize_per_tensor(torch.randn(3, 4, 6), 0.1, 0, torch.qint8),
        "dim": 0,
        "index": torch.tensor([0, 2], dtype=torch.int32),
    },
    {
        "input": torch.quantize_per_tensor(torch.randn(3, 4, 6), 0.1, 1, torch.qint8),
        "dim": 2,
        "index": torch.tensor([1, 3], dtype=torch.int32),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_index_select)
def test_index_select(input_data):
    function(input_data, torch.index_select)


input_masked_fill = [
    {
        "input": torch.quantize_per_tensor(
            torch.randn(3, 4, 6), 0.1, 127, torch.quint8
        ),
        "mask": torch.randint(0, 2, size=(3, 4, 6), dtype=torch.bool),
        "value": 0,
    },
    {
        "input": torch.quantize_per_tensor(torch.randn(3, 4, 6), 0.1, 0, torch.qint8),
        "mask": torch.randint(0, 2, size=(3, 4, 6), dtype=torch.bool),
        "value": 1.1,
    },
    {
        "input": torch.quantize_per_tensor(torch.randn(3, 4, 6), 0.1, 1, torch.qint8),
        "mask": torch.randint(0, 2, size=(1,), dtype=torch.bool),
        "value": 1.1,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_masked_fill)
def test_mask_fill_(input_data):
    test = testing.OpTest(
        func=torch.Tensor.masked_fill_,
        input_args={},
        comparators=testing.QuantizedComparator(),
    )
    test.check_result(list(input_data.values()))


input_unfold = [
    {
        "input": torch.quantize_per_tensor(
            torch.randn(2, 4, 6), 0.1, 127, torch.quint8
        ),
        "dimension": 0,
        "size": 2,
        "step": 1,
    },
    {
        "input": torch.quantize_per_tensor(torch.randn(2, 4, 6), 0.1, 0, torch.qint8),
        "dimension": 0,
        "size": 2,
        "step": 1,
    },
    {
        "input": torch.quantize_per_tensor(torch.randn(2, 4, 6), 0.1, 1, torch.qint8),
        "dimension": 1,
        "size": 2,
        "step": 2,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_unfold)
def test_unfold(input_data):
    test = testing.OpTest(
        func=torch.Tensor.unfold,
        input_args={},
        comparators=testing.QuantizedComparator(),
    )
    test.check_result(
        [
            input_data["input"],
            input_data["dimension"],
            input_data["size"],
            input_data["step"],
        ]
    )


input_pool = [
    {"input": torch.quantize_per_tensor(torch.randn(1, 3, 8, 8), 0.01, 0, torch.qint8)},
    {"input": torch.quantize_per_tensor(torch.randn(1, 3, 8, 8), 0.01, 0, torch.qint8)},
    {"input": torch.quantize_per_tensor(torch.randn(1, 3, 8, 8), 0.01, 0, torch.qint8)},
]
input_pool_size = [
    {"output_size": (4, 4)},
    {"output_size": (4, 1)},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_pool)
@pytest.mark.parametrize("input_args", input_pool_size)
def test_adaptive_avg_pool2d(input_data, input_args):
    test = testing.OpTest(
        func=torch.nn.AdaptiveAvgPool2d,
        input_args=input_args,
        # QuantizedComparator will compare quantized value(int)
        # so error is at least 1 if error exists
        comparators=testing.QuantizedComparator(abs_diff=1),
    )
    test.check_result(input_data)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_pool)
@pytest.mark.parametrize("input_args", input_pool_size)
def test_max_pool2d(input_data, input_args):
    test = testing.OpTest(
        func=torch.quantized_max_pool2d,
        input_args={
            "input": input_data["input"],
            "kernel_size": input_args["output_size"],
        },
        comparators=testing.QuantizedComparator(),
    )
    test.check_result()


input_concat = [
    {
        "tensors": (
            torch.quantize_per_tensor(torch.randn(2, 4, 6), 0.05, 0, torch.qint8),
            torch.quantize_per_tensor(torch.randn(2, 4, 6), 0.141, 0, torch.qint8),
        ),
        "dim": 0,
    },
    {
        "tensors": (
            torch.quantize_per_tensor(torch.randn(2, 4, 6), 0.02, 0, torch.qint8),
            torch.quantize_per_tensor(torch.randn(2, 4, 6), 0.04, 0, torch.qint8),
        ),
        "dim": 1,
    },
    {
        "tensors": (
            torch.quantize_per_tensor(torch.randn(2, 4, 6), 0.05, 0, torch.qint8),
            torch.quantize_per_tensor(torch.randn(2, 4, 6), 0.03, 0, torch.qint8),
        ),
        "dim": 2,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_concat)
def test_concat(input_data):
    function(input_data, torch.cat)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_concat)
def test_qconcat(input_data):
    m = nnq.QFunctional()
    inputs = {
        "x": input_data["tensors"],
        "dim": input_data["dim"],
    }
    function(inputs, m.cat)


input_upsample = [
    {
        "input": torch.quantize_per_tensor(
            torch.randn(1, 3, 4, 4), 0.01, 127, torch.quint8
        )
    },
    {"input": torch.quantize_per_tensor(torch.randn(1, 3, 8, 8), 0.01, 0, torch.qint8)},
    {"input": torch.quantize_per_tensor(torch.randn(1, 3, 2, 2), 0.01, 1, torch.qint8)},
]
input_scale_factor = [
    {"scale_factor": (2, 2)},
    {"scale_factor": (4, 4)},
    {"scale_factor": (8, 8)},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_upsample)
@pytest.mark.parametrize("input_args", input_scale_factor)
def test_upsample_nearest2d(input_data, input_args):
    test = testing.OpTest(
        func=torch.nn.UpsamplingNearest2d,
        input_args=input_args,
        comparators=testing.QuantizedComparator(),
    )
    test.check_result(input_data)


configs = [
    ((2, 2), 0, 255, 1),
    ((2, 3, 16), 0, 255, 1),
    ((1, 4, 8, 8), 0, 127, 1),
    ((2, 2, 32), 10, 240, 1),
]


def get_input_tensor(shape, dtype):
    return torch.randn(shape, dtype=dtype)


def get_scale_tensor(shape, axis, dtype):
    channel_dim = shape[axis]
    return torch.rand(channel_dim, dtype=dtype)


def get_zero_point_tensor(shape, axis, dtype):
    channel_dim = shape[axis]
    return torch.randint(0, 10, (channel_dim,), dtype=dtype)


def get_intinput_tensor(shape, dtype):
    return torch.randint(0, 10, shape, dtype=dtype)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
def test_fake_quantize_per_channel_affine_cachemask(config):
    shape, qmin, qmax, axis = config

    def fake_quant_cachemask(
        input_tensor, scale, zero_point, axis, quant_min, quant_max
    ):
        return torch.ops.aten.fake_quantize_per_channel_affine_cachemask(
            input_tensor, scale, zero_point, axis, quant_min, quant_max
        )

    input_tensor = get_input_tensor(shape, torch.float32)
    scale_tensor = get_scale_tensor(shape, axis, torch.float32)
    zero_point_tensor = get_zero_point_tensor(shape, axis, torch.float32)

    test = testing.OpTest(
        func=fake_quant_cachemask,
        input_args={
            "input_tensor": input_tensor,
            "scale": scale_tensor,
            "zero_point": zero_point_tensor,
            "axis": axis,
            "quant_min": qmin,
            "quant_max": qmax,
        },
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()


configs = [
    ((2, 2), 0, 255),
    ((2, 3, 16), 0, 255),
    ((1, 4, 8, 8), 0, 127),
    ((2, 2, 32), 10, 240),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
def test_fake_quantize_per_tensor_affine_cachemask(config):
    shape, qmin, qmax = config

    def fake_quant_cachemask(
        input_tensor, scale, zero_point, fake_quant_enabled, quant_min, quant_max
    ):
        return torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
            input_tensor, scale, zero_point, fake_quant_enabled, quant_min, quant_max
        )

    input_tensor = get_input_tensor(shape, torch.float32)
    scale_tensor = torch.tensor(0.1, dtype=torch.float32)
    zero_point_tensor = torch.tensor(0, dtype=torch.int)
    fake_quant_enabled_tensor = torch.tensor(1, dtype=torch.long)  #

    test = testing.OpTest(
        func=fake_quant_cachemask,
        input_args={
            "input_tensor": input_tensor,
            "scale": scale_tensor,
            "zero_point": zero_point_tensor,
            "fake_quant_enabled": fake_quant_enabled_tensor,
            "quant_min": qmin,
            "quant_max": qmax,
        },
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )

    test.check_result()


configs = [
    ((4, 8), 0, 255, 0, True, False),  # shape, qmin, qmax, ch_axis, per_row, symmetric
    ((2, 3, 16), -128, 127, 0, True, True),
    ((10,), 0, 255, -1, False, False),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
def test_fused_moving_avg_obs_fq_helper(config):
    shape, qmin, qmax, ch_axis, per_row_fq, symmetric_quant = config

    def fused_obs_fq_helper(
        x,
        observer_on,
        fake_quant_on,
        running_min,
        running_max,
        scale,
        zero_point,
        averaging_const,
        quant_min,
        quant_max,
        ch_axis,
        per_row_fq,
        symmetric_quant,
    ):
        return torch._fused_moving_avg_obs_fq_helper(
            x,
            observer_on,
            fake_quant_on,
            running_min,
            running_max,
            scale,
            zero_point,
            averaging_const,
            quant_min,
            quant_max,
            ch_axis,
            per_row_fq,
            symmetric_quant,
        )

    x = get_input_tensor(shape, torch.float32)

    observer_on = torch.tensor(1, dtype=torch.long)
    fake_quant_on = torch.tensor(1, dtype=torch.long)

    if ch_axis >= 0:
        num_channels = shape[ch_axis]
        running_min = torch.zeros(num_channels, dtype=torch.float32)
        running_max = torch.zeros(num_channels, dtype=torch.float32)
        scale = torch.ones(num_channels, dtype=torch.float32) * 0.1
        zero_point = torch.zeros(num_channels, dtype=torch.int)
    else:
        running_min = torch.tensor([0.0], dtype=torch.float32)
        running_max = torch.tensor([0.0], dtype=torch.float32)
        scale = torch.tensor([0.1], dtype=torch.float32)
        zero_point = torch.tensor([0.0], dtype=torch.int)

    averaging_const = 0.01

    test = testing.OpTest(
        func=fused_obs_fq_helper,
        input_args={
            "x": x,
            "observer_on": observer_on,
            "fake_quant_on": fake_quant_on,
            "running_min": running_min,
            "running_max": running_max,
            "scale": scale,
            "zero_point": zero_point,
            "averaging_const": averaging_const,
            "quant_min": qmin,
            "quant_max": qmax,
            "ch_axis": ch_axis,
            "per_row_fq": per_row_fq,
            "symmetric_quant": symmetric_quant,
        },
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )

    test.check_result()


configs = [
    ((1, 10), (10, 5)),
    ((2, 3), (3, 4)),
    ((10, 7), (7, 10)),
]


@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_run_linear_vs_onednn(config, dtype):
    shape1, shape2 = config
    test = testing.OpTest(
        func=torch.ops.quantized.linear_per_tensor,
        refer_func=torch.ops.onednn.qlinear_pointwise.default,
        input_args={},
    )

    act = get_intinput_tensor(shape1, torch.int8)
    weight = get_intinput_tensor(shape2, torch.int8)

    act_scale = 1.0
    act_zero_point = 0
    weight_scale = torch.randint(0, 10, [shape2[1]], dtype=torch.float32)
    weight_zero_point = torch.zeros([shape2[1]], dtype=torch.int)

    bias = torch.ones([shape2[1]], dtype=torch.float32)
    output_scale = 1.0
    output_zero_point = 0

    input_args = {
        "qx": act,
        "x_scale": act_scale,
        "x_zero_point": act_zero_point,
        "qw": weight,
        "w_scale": weight_scale,
        "w_zero_point": weight_zero_point,
        "bias": bias,
        "output_scale": output_scale,
        "output_zero_point": output_zero_point,
        "output_dtype": dtype,
        "post_op_name": "none",
        "post_op_args": [],
        "post_op_algorithm": "",
    }

    weight_mlknn = weight.to_mkldnn()

    cpu_args = {**input_args, "qw": weight_mlknn}

    out_musa = test._call_func(inputs=input_args, device="musa", refer=False)
    out_cpu = test._call_func(
        inputs=cpu_args, device="cpu", refer=True, onednn_flag=True
    )
    test.compare_res(out_musa, out_cpu)


configs = [
    ((1, 1, 5, 5), (1, 1, 3, 3)),
    ((1, 3, 10, 10), (4, 3, 2, 2)),
    ((10, 5, 6, 6), (8, 5, 2, 2)),
]


@pytest.mark.parametrize(
    "dtype, attr",
    [
        (torch.float32, "none"),
        (torch.float32, "relu"),
    ],
)
@pytest.mark.parametrize("config", configs)
def test_run_pointwise_vs_onednn(config, dtype, attr):
    shape1, shape2 = config
    act = get_intinput_tensor(shape1, dtype=torch.uint8)
    weight = get_intinput_tensor(shape2, dtype=torch.int8)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1

    act_scale = 1.0
    act_zero_point = 0

    weight_scale = torch.randn([shape2[0]], dtype=torch.float)
    weight_zero_point = torch.zeros([shape2[0]], dtype=torch.int)

    output_scale = 1.0
    output_zero_point = 0

    bias = torch.randn([shape2[0]])

    test = testing.OpTest(
        func=torch.ops.quantized.conv,
        refer_func=torch.ops.onednn.qconv2d_pointwise,
        input_args={},
        comparators=testing.AbsDiffComparator(1),
    )

    input_args = {
        "qx": act,
        "x_scale": act_scale,
        "x_zero_point": act_zero_point,
        "qw": weight,
        "w_scale": weight_scale,
        "w_zero_point": weight_zero_point,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "output_scale": output_scale,
        "output_zero_point": output_zero_point,
        "output_dtype": dtype,
        "attr": attr,
        "scalars": [],
        "algorithm": "",
    }
    weight = weight.to_mkldnn()
    cpu_input_args = {**input_args, "qw": weight}
    torch.set_printoptions(threshold=10000)

    out_musa = test._call_func(inputs=input_args, device="musa", refer=False)
    out_cpu = test._call_func(
        inputs=cpu_input_args, device="cpu", refer=True, onednn_flag=True
    )
    test.compare_res(out_musa, out_cpu)


configs = [((1, 1, 5, 5), (1, 1, 2, 2)), ((1, 1, 6, 6), (1, 1, 2, 2))]


@pytest.mark.parametrize(
    "dtype, attr",
    [
        (torch.float32, "relu"),
        (torch.uint8, "relu"),
        (torch.uint8, "none"),
        (torch.int8, "relu"),
    ],
)
@pytest.mark.parametrize("config", configs)
def test_run_pointwise_binary_vs_onednn(config, dtype, attr):
    shape1, shape2 = config
    act = get_intinput_tensor(shape1, dtype=torch.uint8)
    weight = get_intinput_tensor(shape2, dtype=torch.int8)

    act_scale = 3.0
    act_zero_point = 0

    weight_scale = torch.randn([shape2[0]], dtype=torch.float32)
    weight_zero_point = torch.zeros([shape2[0]], dtype=torch.int)
    H_out = shape1[2] - shape2[2] + 1
    W_out = shape1[3] - shape2[3] + 1
    accum = torch.randint(3, 10, (shape1[0], shape2[0], H_out, W_out), dtype=dtype)

    accum_scale = 1.0
    accum_zero_point = 0

    output_scale = 1.0
    output_zero_point = 0

    bias = torch.ones([shape2[0]])

    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1

    test = testing.OpTest(
        func=torch.ops.quantized.conv_binary,
        refer_func=torch.ops.onednn.qconv2d_pointwise.binary,
        input_args={},
    )

    input_args = {
        "qx": act,
        "x_scale": act_scale,
        "x_zero_point": act_zero_point,
        "qw": weight,
        "w_scale": weight_scale,
        "w_zero_point": weight_zero_point,
        "qaccum": accum,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "output_scale": output_scale,
        "output_zero_point": output_zero_point,
        "output_dtype": dtype,
        "accum_scale": accum_scale,
        "accum_zero_point": accum_zero_point,
        "binary_attr": "sum",
        "alpha": 1.0,
        "unary_attr": attr,
        "unary_scalars": [],
        "unary_algorithm": "",
    }

    weight = weight.to_mkldnn()
    cpu_input_args = {**input_args, "qw": weight}

    out_musa = test._call_func(inputs=input_args, device="musa", refer=False)
    out_cpu = test._call_func(
        inputs=cpu_input_args, device="cpu", refer=True, onednn_flag=True
    )
    test.compare_res(out_musa, out_cpu)
