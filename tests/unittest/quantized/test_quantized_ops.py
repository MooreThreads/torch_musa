"""Test quantized operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
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
    {
        "input": torch.quantize_per_tensor(
            torch.randn(1, 3, 8, 8), 0.01, 0, torch.qint8
        )
    },
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
