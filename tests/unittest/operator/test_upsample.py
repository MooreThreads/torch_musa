"""Test upsampling operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from functools import partial
import pytest
import torch
from torch_musa import testing


all_support_types = [torch.float32]
scale_factor = [2, 1]
# TODO(@mt-ai): upsample bilinear with antialias=True would hang
antialiases = [False]


def function(input_data, dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test.check_grad_fn()


def function_fp16(input_data, dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-2),
    )
    test.check_musafp16_vs_musafp32()
    test.check_grad_fn(fp16=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([2, 1, 10, 10])},
        {"input": torch.randn([10, 6, 256, 300])},
        {"input": torch.randn([4, 3, 228, 304])},
        {"input": torch.randn([4, 16, 32, 32])},
        {"input": torch.randn([4, 16, 32, 32]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([4, 1, 32, 32]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([4, 16, 2, 2]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([0, 16, 2, 2]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([0, 16, 2, 2])},
    ],
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("antialias", antialiases)
def test_upsample_bilinear(input_data, dtype, scale_factor, align_corners, antialias):
    bilinear = partial(
        torch.nn.functional.interpolate,
        mode="bilinear",
        scale_factor=scale_factor,
        align_corners=align_corners,
        antialias=antialias,
    )
    function(input_data, dtype, bilinear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([2, 1, 10, 10])},
        {"input": torch.randn([10, 6, 256, 300])},
        {"input": torch.randn([4, 3, 228, 304])},
        {"input": torch.randn([4, 16, 32, 32])},
        {"input": torch.randn([4, 16, 32, 32]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([4, 1, 32, 32]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([4, 16, 1, 1]).to(memory_format=torch.channels_last)},
    ],
)
@pytest.mark.parametrize("dtype", all_support_types + [torch.uint8])
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_nearest2d(input_data, dtype, scale_factor):
    nearest = partial(
        torch.nn.functional.interpolate, mode="nearest", scale_factor=scale_factor
    )
    function(input_data, dtype, nearest)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        torch.randn([2, 1, 10, 10]),
        torch.randn([10, 6, 256, 300]),
        torch.randn([4, 3, 228, 304]),
        torch.randn([4, 16, 32, 32]),
        torch.randn([4, 16, 32, 32]).to(memory_format=torch.channels_last),
        torch.randn([4, 1, 32, 32]).to(memory_format=torch.channels_last),
        torch.randn([4, 16, 1, 1]).to(memory_format=torch.channels_last),
    ],
)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("mode", ["nearest-exact"])
def test_upsample_nearest2d_bwd(input_data, scale_factor, mode):
    model = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
    model.train(True)

    input_cpu = input_data.cpu()
    add_cpu = input_cpu + 0.0001
    add_cpu.requires_grad_()
    output_cpu = model(add_cpu)
    output_cpu.sum().backward()

    input_musa = input_data.to("musa")
    add_musa = input_musa + 0.0001
    add_musa.requires_grad_()
    output_musa = model(add_musa)
    output_musa.sum().backward()

    assert testing.DefaultComparator(abs_diff=1e-5)(add_cpu.grad, add_musa.grad.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([2, 1, 10, 10])},
        {"input": torch.randn([10, 6, 256, 300])},
        {"input": torch.randn([4, 3, 228, 304])},
        {"input": torch.randn([4, 16, 32, 32])},
        {"input": torch.randn([4, 16, 32, 32]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([4, 1, 32, 32]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([4, 16, 1, 1]).to(memory_format=torch.channels_last)},
    ],
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="fp16 upsample nearest supported in QY2 or later",
)
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_nearest2d_fp16(input_data, dtype, scale_factor):
    nearest = partial(
        torch.nn.functional.interpolate, mode="nearest", scale_factor=scale_factor
    )
    function_fp16(input_data, dtype, nearest)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([2, 10, 10, 1])},
        {"input": torch.randn([10, 256, 300, 6])},
        {"input": torch.randn([4, 228, 304, 3])},
        {"input": torch.randn([4, 16, 32, 32]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([4, 1, 32, 32]).to(memory_format=torch.channels_last)},
        {"input": torch.randn([4, 16, 1, 1]).to(memory_format=torch.channels_last)},
    ],
)
@pytest.mark.parametrize("dtype", all_support_types + [torch.uint8])
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_nearest2d_uncontig(input_data, dtype, scale_factor):
    nearest = partial(
        torch.nn.functional.interpolate, mode="nearest", scale_factor=scale_factor
    )
    nchw_tensor = input_data["input"].permute(0, 3, 1, 2)
    function({"input": nchw_tensor}, dtype, nearest)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([2, 10, 10])},
        {"input": torch.randn([10, 256, 300])},
        {"input": torch.randn([4, 228, 304])},
        {"input": torch.randn([4, 32, 32])},
    ],
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_linear(input_data, dtype, scale_factor, align_corners):
    linear = partial(
        torch.nn.functional.interpolate,
        mode="linear",
        scale_factor=scale_factor,
        align_corners=align_corners,
    )
    function(input_data, dtype, linear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([2, 1, 1, 10, 10])},
        {"input": torch.randn([10, 8, 6, 256, 300])},
        {"input": torch.randn([4, 9, 3, 228, 304])},
        {"input": torch.randn([4, 25, 16, 32, 32])},
        {"input": torch.randn([0, 3, 4, 5, 6])},
        {
            "input": torch.randn([2, 1, 1, 10, 10]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {
            "input": torch.randn([10, 8, 6, 256, 300]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {
            "input": torch.randn([4, 9, 3, 228, 304]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {
            "input": torch.randn([4, 25, 16, 32, 32]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {
            "input": torch.randn([0, 3, 4, 5, 6]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {"input": torch.randn([4, 25, 16, 32, 32])[:, :10, :, :, :]},
        {"input": torch.randn([4, 25, 16, 32, 32])[:, :10, :10, :, :]},
        {"input": torch.randn([4, 25, 16, 32, 32])[:, :10, :10, :10, :]},
        {"input": torch.randn([4, 25, 16, 32, 32])[:, :10, :10, :10, :10]},
    ],
)
@pytest.mark.parametrize("dtype", all_support_types + [torch.uint8])
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_nearest3d(input_data, dtype, scale_factor):
    linear = partial(
        torch.nn.functional.interpolate, mode="nearest", scale_factor=scale_factor
    )
    function(input_data, dtype, linear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        torch.randn([2, 1, 1, 10, 10]),
        torch.randn([10, 8, 6, 256, 300]),
        torch.randn([4, 9, 3, 228, 304]),
        torch.randn([4, 25, 16, 32, 32]),
        torch.randn([0, 3, 4, 5, 6]),
        torch.randn([2, 1, 1, 10, 10]).to(memory_format=torch.channels_last_3d),
        torch.randn([10, 8, 6, 256, 300]).to(memory_format=torch.channels_last_3d),
        torch.randn([4, 9, 3, 228, 304]).to(memory_format=torch.channels_last_3d),
        torch.randn([4, 25, 16, 32, 32]).to(memory_format=torch.channels_last_3d),
        torch.randn([0, 3, 4, 5, 6]).to(memory_format=torch.channels_last_3d),
    ],
)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("mode", ["nearest", "nearest-exact"])
def test_upsample_nearest3d_bwd(input_data, scale_factor, mode):
    model = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
    model.train(True)

    input_cpu = input_data.cpu()
    add_cpu = input_cpu + 0.0001
    add_cpu.requires_grad_()
    output_cpu = model(add_cpu)
    output_cpu.sum().backward()

    input_musa = input_data.to("musa")
    add_musa = input_musa + 0.0001
    add_musa.requires_grad_()
    output_musa = model(add_musa)
    output_musa.sum().backward()

    assert testing.DefaultComparator(abs_diff=1e-5)(add_cpu.grad, add_musa.grad.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([2, 1, 1])},
        {"input": torch.randn([10, 8, 6])},
        {"input": torch.randn([4, 9, 3])},
        {"input": torch.randn([4, 25, 16])},
    ],
)
@pytest.mark.parametrize("dtype", all_support_types + [torch.uint8])
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_nearest1d(input_data, dtype, scale_factor):
    linear = partial(
        torch.nn.functional.interpolate, mode="nearest", scale_factor=scale_factor
    )
    function(input_data, dtype, linear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        torch.randn([2, 1, 1]),
        torch.randn([10, 8, 6]),
        torch.randn([4, 9, 3]),
        torch.randn([4, 25, 16]),
    ],
)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("mode", ["nearest-exact"])
def test_upsample_nearest1d_bwd(input_data, scale_factor, mode):
    model = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
    model.train(True)

    input_cpu = input_data.cpu()
    add_cpu = input_cpu + 0.0001
    add_cpu.requires_grad_()
    output_cpu = model(add_cpu)
    output_cpu.sum().backward()

    input_musa = input_data.to("musa")
    add_musa = input_musa + 0.0001
    add_musa.requires_grad_()
    output_musa = model(add_musa)
    output_musa.sum().backward()

    assert testing.DefaultComparator(abs_diff=1e-5)(add_cpu.grad, add_musa.grad.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randn([2, 1, 1, 10, 10])},
        {"input": torch.randn([10, 8, 6, 256, 300])},
        {"input": torch.randn([4, 9, 3, 228, 304])},
        {"input": torch.randn([4, 25, 16, 32, 32])},
        {"input": torch.randn([0, 3, 4, 5, 6])},
        {
            "input": torch.randn([2, 1, 1, 10, 10]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {
            "input": torch.randn([10, 8, 6, 256, 300]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {
            "input": torch.randn([4, 9, 3, 228, 304]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {
            "input": torch.randn([4, 25, 16, 32, 32]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {
            "input": torch.randn([0, 3, 4, 5, 6]).to(
                memory_format=torch.channels_last_3d
            )
        },
        {"input": torch.randn([4, 25, 16, 32, 32])[:, :10, :, :, :]},
        {"input": torch.randn([4, 25, 16, 32, 32])[:, :10, :10, :, :]},
        {"input": torch.randn([4, 25, 16, 32, 32])[:, :10, :10, :10, :]},
        {"input": torch.randn([4, 25, 16, 32, 32])[:, :10, :10, :10, :10]},
    ],
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_trilinear3d(input_data, dtype, scale_factor):
    linear = partial(
        torch.nn.functional.interpolate, mode="trilinear", scale_factor=scale_factor
    )
    function(input_data, dtype, linear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        torch.randn([2, 1, 1, 10, 10]),
        torch.randn([10, 8, 6, 256, 300]),
        torch.randn([4, 9, 3, 228, 304]),
        torch.randn([4, 25, 16, 32, 32]),
        torch.randn([0, 3, 4, 5, 6]),
        torch.randn([2, 1, 1, 10, 10]).to(memory_format=torch.channels_last_3d),
        torch.randn([10, 8, 6, 256, 300]).to(memory_format=torch.channels_last_3d),
        torch.randn([4, 9, 3, 228, 304]).to(memory_format=torch.channels_last_3d),
        torch.randn([4, 25, 16, 32, 32]).to(memory_format=torch.channels_last_3d),
        torch.randn([0, 3, 4, 5, 6]).to(memory_format=torch.channels_last_3d),
    ],
)
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_trilinear3d_bwd(input_data, scale_factor):
    model = torch.nn.Upsample(scale_factor=scale_factor, mode="trilinear")
    model.train(True)

    input_cpu = input_data.cpu()
    add_cpu = input_cpu + 0.0001
    add_cpu.requires_grad_()
    output_cpu = model(add_cpu)
    output_cpu.sum().backward()

    input_musa = input_data.to("musa")
    add_musa = input_musa + 0.0001
    add_musa.requires_grad_()
    output_musa = model(add_musa)
    output_musa.sum().backward()

    assert testing.DefaultComparator(abs_diff=1e-5)(add_cpu.grad, add_musa.grad.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config",
    [
        # input_shape, scale_factor, align_corners, is_channels_last
        [(2, 8, 32, 32), (1.5, 1.5), False, False],
        [(2, 8, 32, 32), (1.5, 1.5), False, True],
        [(2, 8, 32, 32), (0.5, 0.5), True, False],
        [(2, 8, 32, 32), (0.5, 0.5), True, True],
    ],
)
@pytest.mark.parametrize("dtype", testing.get_float_types())
def test_upsample_bicubic_aa(config, dtype):
    tensor = torch.randn(config[0], dtype=dtype)
    if config[3]:
        memory_format = (
            torch.channels_last if len(config[0]) == 4 else torch.channels_last_3d
        )
        tensor = tensor.to(memory_format=memory_format)
    tensor.requires_grad_(True)
    func = partial(
        torch.nn.functional.interpolate,
        mode="bicubic",
        scale_factor=config[1],
        align_corners=config[2],
        antialias=True,
    )
    input_args = {"input": tensor}

    test = testing.OpTest(
        func=func,
        input_args=input_args,
        comparators=testing.DefaultComparator(abs_diff=5e-2, rel_diff=1e-2),
    )
    if dtype == torch.half:
        test.check_musafp16_vs_musafp32(train=True)
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16(train=True)
    else:
        test.check_result(train=True)
