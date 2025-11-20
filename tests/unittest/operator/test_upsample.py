"""Test upsampling operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from functools import partial
import pytest
import torch
from torch_musa import testing


all_support_types = [torch.float32]
if testing.get_musa_arch() >= 22:
    all_support_types += [torch.float16, torch.bfloat16]
scale_factor = [2, 3]


upsample1d_configs = [
    # shape, memory_format
    [[2, 10, 10], None],
    [[10, 256, 300], None],
    [[4, 228, 304], None],
    [[2, 32, 32], None],
]
upsample2d_configs = [
    # shape, memory_format
    [[2, 1, 10, 10], None],
    [[4, 16, 32, 32], None],
    [[4, 16, 32, 32], torch.channels_last],
    [[4, 16, 1, 1], torch.channels_last],
    [[0, 16, 1, 1], None],
]
upsample3d_configs = [
    # shape, memory_format
    [[2, 1, 1, 10, 10], None],
    [[4, 25, 16, 32, 32], None],
    [[0, 3, 4, 5, 6], None],
    [[4, 25, 16, 32, 32], torch.channels_last_3d],
    [[0, 3, 4, 5, 6], torch.channels_last_3d],
    [[4, 25, 16, 32, 32], None],
]


def get_upsample_diff(dtype):
    if dtype in [torch.float16, torch.bfloat16]:
        abs_diff = 1e-2
        rel_diff = 1e-2
    else:
        abs_diff = 1e-5
        rel_diff = 1e-5
    return abs_diff, rel_diff


def function(input_data, dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    abs_diff, rel_diff = get_upsample_diff(dtype)
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    if dtype in [torch.float16, torch.bfloat16]:
        test.check_musafp16_vs_musafp32()
        test.check_grad_fn(fp16=True)
    else:
        test.check_result()
        test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", upsample1d_configs)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_linear(config, dtype, scale_factor, align_corners):
    shape, memory_format = config
    if memory_format is not None:
        input_data = {"input": torch.randn(shape).to(memory_format=memory_format)}
    else:
        input_data = {"input": torch.randn(shape)}
    linear = partial(
        torch.nn.functional.interpolate,
        mode="linear",
        scale_factor=scale_factor,
        align_corners=align_corners,
    )
    function(input_data, dtype, linear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", upsample2d_configs)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("antialias", [False, True])
def test_upsample_bilinear(config, dtype, scale_factor, align_corners, antialias):
    shape, memory_format = config
    if memory_format is not None:
        input_data = {"input": torch.randn(shape).to(memory_format=memory_format)}
    else:
        input_data = {"input": torch.randn(shape)}
    bilinear = partial(
        torch.nn.functional.interpolate,
        mode="bilinear",
        scale_factor=scale_factor,
        align_corners=align_corners,
        antialias=antialias,
    )
    function(input_data, dtype, bilinear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", upsample3d_configs)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_trilinear(config, dtype, scale_factor):
    shape, memory_format = config
    if memory_format is not None:
        input_data = {"input": torch.randn(shape).to(memory_format=memory_format)}
    else:
        input_data = {"input": torch.randn(shape)}
    linear = partial(
        torch.nn.functional.interpolate, mode="trilinear", scale_factor=scale_factor
    )
    function(input_data, dtype, linear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config", upsample1d_configs + upsample2d_configs + upsample3d_configs
)
@pytest.mark.parametrize("dtype", all_support_types + [torch.uint8])
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("exact", [True, False])
def test_upsample_nearest(config, dtype, scale_factor, exact):
    shape, memory_format = config
    if memory_format is not None:
        input_data = {"input": torch.randn(shape).to(memory_format=memory_format)}
    else:
        input_data = {"input": torch.randn(shape)}
    linear = partial(
        torch.nn.functional.interpolate,
        mode="nearest-exact" if exact else "nearest",
        scale_factor=scale_factor,
    )
    function(input_data, dtype, linear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", upsample2d_configs)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("antialias", [False, True])
def test_upsample_bicubic(config, dtype, scale_factor, align_corners, antialias):
    shape, memory_format = config
    if memory_format is not None:
        input_data = {"input": torch.randn(shape).to(memory_format=memory_format)}
    else:
        input_data = {"input": torch.randn(shape)}
    bilinear = partial(
        torch.nn.functional.interpolate,
        mode="bicubic",
        scale_factor=scale_factor,
        align_corners=align_corners,
        antialias=antialias,
    )
    function(input_data, dtype, bilinear)
