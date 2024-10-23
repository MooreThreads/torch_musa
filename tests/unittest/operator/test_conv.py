"""Test conv operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name
import torch
import pytest

from torch_musa import testing


def get_conv_dtypes():
    dtypes = [
        torch.float32,
    ]
    if testing.get_musa_arch() >= 22:
        dtypes.extend([torch.float16, torch.bfloat16])
    return dtypes


def get_conv_test_tolerance(direction="FWD"):
    if direction == "FWD":
        return {
            torch.float32: (1e-5, 1e-5),
            torch.float16: (2e-3, 2e-3),
            torch.bfloat16: (2e-2, 2e-2),
        }
    if direction == "BWD":
        return {
            torch.float32: (5e-3, 5e-3),
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (5e-2, 5e-2),
        }
    raise ValueError(f"illegal direction: {direction}")


conv1d_fwd_configs = [
    # sizes, in_c, out_c, k, s, p, d, g
    [(2, 3, 16), 3, 3, 3, 1, 1, 1, 3],  # depthwise
    [(2, 3, 16), 3, 8, 3, 1, 1, 1, 1],
    [(2, 4, 64), 4, 8, 5, 1, 1, 1, 1],
]

conv1d_bwd_configs = [
    # in_sizes, out_sizes, k, s, p, d, g
    [(2, 3, 16), (2, 8, 16), (3,), (1,), (1,), (1,), 1],
    [(2, 7, 64), (2, 64, 32), (3,), (2,), (1,), (1,), 1],
]

conv2d_fwd_configs = [
    # sizes, in_c, out_c, k, s, p, d, g
    [(2, 3, 16, 16), 3, 3, (3, 3), (1, 1), (1, 1), (1, 1), 3],  # depthwise
    # [(1, 256, 8, 8), 256, 256, (3, 3), (1, 1), (1, 1), (1, 1), 1],  # conv3x3s1d1g1
    [(2, 4, 64, 64), 4, 1, (5, 5), (1, 1), (1, 1), (1, 1), 1],
    [(0, 1, 64, 64), 1, 1, (5, 5), (1, 1), (1, 1), (1, 1), 1],
]

conv2d_bwd_configs = [
    # in_sizes, out_sizes, k, s, p, d, g
    [(2, 3, 16, 16), (2, 1, 16, 16), (3, 3), (1, 1), (1, 1), (1, 1), 1],
    [(2, 7, 224, 224), (2, 64, 112, 112), (3, 3), (2, 2), (1, 1), (1, 1), 1],
]

conv3d_fwd_configs = [
    [
        (1, 3, 8, 64, 64),
        3,
        3,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        3,
    ],  # depthwise
    [(1, 3, 8, 64, 64), 3, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1],
    [(4, 3, 8, 32, 32), 3, 64, (3, 7, 7), (2, 2, 2), (3, 3, 3), (1, 1, 1), 1],
    [(0, 3, 8, 224, 224), 3, 64, (3, 7, 7), (2, 2, 2), (3, 3, 3), (1, 1, 1), 1],
]

conv3d_bwd_configs = [
    # in_sizes, out_sizes, k, s, p, d, g
    [
        (1, 3, 8, 64, 64),
        (1, 16, 8, 64, 64),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        1,
    ],
    [
        (4, 3, 8, 32, 32),
        (4, 64, 6, 16, 16),
        (3, 7, 7),
        (2, 2, 2),
        (3, 3, 3),
        (1, 1, 1),
        1,
    ],
]

conv_transpose1d_fwd_configs = [
    # in_sizes, in_c, out_c, k, s, p, d, out_p, g
    [(2, 16, 50), 16, 3, 3, 2, 4, 1, 0, 1],
]

conv_transpose1d_bwd_configs = [
    # in_sizes, out_sizes, k, s, p, d, out_p, g
    [(2, 16, 50), (2, 3, 93), (3,), (2,), (4,), (1,), (0,), 1],
]

conv_transpose2d_fwd_configs = [
    # in_sizes, in_c, out_c, k, s, p, d, out_p, g
    [(20, 16, 50, 100), 16, 3, (3, 5), (2, 1), (4, 2), (1, 1), (0, 0), 1],
]

conv_transpose2d_bwd_configs = [
    # in_sizes, out_sizes, k, s, p, d, out_p, g
    [(20, 16, 50, 100), (20, 3, 93, 100), (3, 5), (2, 1), (4, 2), (1, 1), (0, 0), 1],
]

conv_transpose3d_fwd_configs = [
    # in_sizes, in_c, out_c, k, s, p, d, out_p, g
    [
        (1, 16, 32, 32, 32),
        16,
        3,
        (3, 5, 5),
        (2, 1, 1),
        (4, 2, 2),
        (1, 1, 1),
        (0, 0, 0),
        1,
    ],
]

conv_transpose3d_bwd_configs = [
    # in_sizes, out_sizes, k, s, p, d, out_p, g
    [
        (1, 16, 32, 32, 32),
        (1, 3, 57, 32, 32),
        (3, 5, 5),
        (2, 1, 1),
        (4, 2, 2),
        (1, 1, 1),
        (0, 0, 0),
        1,
    ]
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv1d_fwd_configs)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", get_conv_dtypes())
def test_conv1d_fwd(common_config, bias, dtype):
    conv_args = {
        "in_channels": common_config[1],
        "out_channels": common_config[2],
        "kernel_size": common_config[3],
        "stride": common_config[4],
        "padding": common_config[5],
        "dilation": common_config[6],
        "groups": common_config[7],
        "bias": bias,
    }
    abs_diff, rel_diff = get_conv_test_tolerance("FWD")[dtype]
    func = torch.nn.Conv1d(**conv_args).to(dtype)
    img = torch.randn(common_config[0], dtype=dtype)
    test = testing.OpTest(
        func=func,
        input_args={"input": img},
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32()
    else:
        test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv1d_bwd_configs)
@pytest.mark.parametrize("dtype", get_conv_dtypes())
def test_conv1d_bwd(common_config, dtype):
    weight_shape = [common_config[1][1], common_config[0][1], *common_config[2]]
    input_shape = common_config[0]
    grad_output_shape = common_config[1]

    weight = torch.randn(weight_shape, dtype=dtype)
    input_ = torch.randn(input_shape, dtype=dtype)
    grad_output = torch.randn(grad_output_shape, dtype=dtype)

    abs_diff, rel_diff = get_conv_test_tolerance("BWD")[dtype]
    test = testing.OpTest(
        func=torch.ops.aten.convolution_backward,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    inputs = [
        grad_output,
        input_,
        weight,
        None,
        common_config[3],  # stride
        common_config[4],  # padding
        common_config[5],  # dilation
        False,
        [
            0,
        ],
        common_config[6],  # groups
        [True, True, False],
    ]

    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32(inputs=inputs)
    else:
        test.check_result(inputs=inputs)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv2d_fwd_configs)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("channels_last", [True, False])
@pytest.mark.parametrize("dtype", get_conv_dtypes())
def test_conv2d_fwd(common_config, bias, channels_last, dtype):
    conv2d_args = {
        "in_channels": common_config[1],
        "out_channels": common_config[2],
        "kernel_size": common_config[3],
        "stride": common_config[4],
        "padding": common_config[5],
        "dilation": common_config[6],
        "groups": common_config[7],
        "bias": bias,
    }
    abs_diff, rel_diff = get_conv_test_tolerance("FWD")[dtype]
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    func = torch.nn.Conv2d(**conv2d_args).to(memory_format=memory_format)
    img = torch.randn(common_config[0]).type(dtype)
    test = testing.OpTest(
        func=func,
        input_args={"input": img},
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        test_dtype=dtype,
    )
    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32()
    else:
        test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv2d_bwd_configs)
@pytest.mark.parametrize("dtype", get_conv_dtypes())
@pytest.mark.parametrize("channels_last", [True, False])
def test_conv2d_bwd(common_config, dtype, channels_last):
    weight_shape = [common_config[1][1], common_config[0][1], *common_config[2]]
    input_shape = common_config[0]
    grad_output_shape = common_config[1]

    memory_format = torch.channels_last if channels_last else torch.contiguous_format

    weight = torch.randn(weight_shape, dtype=dtype).to(memory_format=memory_format)
    input_ = torch.randn(input_shape, dtype=dtype)
    grad_output = torch.randn(grad_output_shape, dtype=dtype)

    abs_diff, rel_diff = get_conv_test_tolerance("BWD")[dtype]
    test = testing.OpTest(
        func=torch.ops.aten.convolution_backward,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    inputs = [
        grad_output,
        input_,
        weight,
        None,
        common_config[3],  # stride
        common_config[4],  # padding
        common_config[5],  # dilation
        False,
        [0, 0],
        common_config[6],
        [False, True, False],
    ]

    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32(inputs=inputs)
    else:
        test.check_result(inputs=inputs)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv3d_fwd_configs)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("channels_last", [True, False])
@pytest.mark.parametrize("dtype", get_conv_dtypes())
def test_conv3d_fwd(common_config, bias, channels_last, dtype):
    conv_args = {
        "in_channels": common_config[1],
        "out_channels": common_config[2],
        "kernel_size": common_config[3],
        "stride": common_config[4],
        "padding": common_config[5],
        "dilation": common_config[6],
        "groups": common_config[7],
        "bias": bias,
    }
    abs_diff, rel_diff = get_conv_test_tolerance("FWD")[dtype]
    memory_format = torch.channels_last_3d if channels_last else torch.contiguous_format
    func = torch.nn.Conv3d(**conv_args).to(memory_format=memory_format)
    test = testing.OpTest(
        func=func,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        test_dtype=dtype,
    )
    img = torch.randn(common_config[0]).type(dtype)
    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32({"input": img})
    else:
        test.check_result({"input": img})


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv3d_bwd_configs)
@pytest.mark.parametrize("dtype", get_conv_dtypes())
@pytest.mark.parametrize("channels_last", [True, False])
def test_conv3d_bwd(common_config, dtype, channels_last):
    weight_shape = [common_config[1][1], common_config[0][1], *common_config[2]]
    input_shape = common_config[0]
    grad_output_shape = common_config[1]

    memory_format = torch.channels_last_3d if channels_last else torch.contiguous_format

    weight = torch.randn(weight_shape, dtype=dtype).to(memory_format=memory_format)
    input_ = torch.randn(input_shape, dtype=dtype)
    grad_output = torch.randn(grad_output_shape, dtype=dtype)

    abs_diff, rel_diff = get_conv_test_tolerance("BWD")[dtype]
    test = testing.OpTest(
        func=torch.ops.aten.convolution_backward,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    inputs = [
        grad_output,
        input_,
        weight,
        None,
        common_config[3],  # stride
        common_config[4],  # padding
        common_config[5],  # dilation
        False,
        [0, 0, 0],  # output_padding
        common_config[6],
        [False, True, False],
    ]

    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32(inputs=inputs)
    else:
        test.check_result(inputs=inputs)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv_transpose1d_fwd_configs)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", get_conv_dtypes())
def test_conv_transpose1d_fwd(common_config, bias, dtype):
    conv_args = {
        "in_channels": common_config[1],
        "out_channels": common_config[2],
        "kernel_size": common_config[3],
        "stride": common_config[4],
        "padding": common_config[5],
        "dilation": common_config[6],
        "output_padding": common_config[7],
        "groups": common_config[8],
        "bias": bias,
    }
    abs_diff, rel_diff = get_conv_test_tolerance("FWD")[dtype]
    test = testing.OpTest(
        func=torch.nn.ConvTranspose1d,
        input_args=conv_args,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        test_dtype=dtype,
    )
    img = torch.randn(common_config[0]).type(dtype)
    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32({"input": img})
    else:
        test.check_result({"input": img})


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv_transpose1d_bwd_configs)
@pytest.mark.parametrize("dtype", get_conv_dtypes())
def test_conv_transpose1d_bwd(common_config, dtype):
    weight_shape = [common_config[0][1], common_config[1][1], *common_config[2]]
    input_shape = common_config[0]
    grad_output_shape = common_config[1]

    weight = torch.randn(weight_shape, dtype=dtype)
    input_ = torch.randn(input_shape, dtype=dtype)
    grad_output = torch.randn(grad_output_shape, dtype=dtype)

    abs_diff, rel_diff = get_conv_test_tolerance("BWD")[dtype]
    test = testing.OpTest(
        func=torch.ops.aten.convolution_backward,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    inputs = [
        grad_output,
        input_,
        weight,
        None,
        common_config[3],  # stride
        common_config[4],  # padding
        common_config[5],  # dilation
        True,
        common_config[6],
        common_config[7],
        [True, True, False],
    ]

    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32(inputs=inputs)
    else:
        test.check_result(inputs=inputs)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv_transpose2d_fwd_configs)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("channels_last", [True, False])
@pytest.mark.parametrize("dtype", get_conv_dtypes())
def test_conv_transpose2d_fwd(common_config, bias, channels_last, dtype):
    conv_args = {
        "in_channels": common_config[1],
        "out_channels": common_config[2],
        "kernel_size": common_config[3],
        "stride": common_config[4],
        "padding": common_config[5],
        "dilation": common_config[6],
        "output_padding": common_config[7],
        "groups": common_config[8],
        "bias": bias,
    }
    abs_diff, rel_diff = get_conv_test_tolerance("FWD")[dtype]
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    func = torch.nn.ConvTranspose2d(**conv_args).to(memory_format=memory_format)
    test = testing.OpTest(
        func=func,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        test_dtype=dtype,
    )
    img = torch.randn(common_config[0]).type(dtype)
    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32({"input": img})
    else:
        test.check_result({"input": img})


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv_transpose2d_bwd_configs)
@pytest.mark.parametrize("dtype", get_conv_dtypes())
@pytest.mark.parametrize("channels_last", [True, False])
def test_conv_transpose2d_bwd(common_config, dtype, channels_last):
    weight_shape = [common_config[0][1], common_config[1][1], *common_config[2]]
    input_shape = common_config[0]
    grad_output_shape = common_config[1]

    memory_format = torch.channels_last if channels_last else torch.contiguous_format

    weight = torch.randn(weight_shape, dtype=dtype).to(memory_format=memory_format)
    input_ = torch.randn(input_shape, dtype=dtype)
    grad_output = torch.randn(grad_output_shape, dtype=dtype)

    abs_diff, rel_diff = get_conv_test_tolerance("BWD")[dtype]
    test = testing.OpTest(
        func=torch.ops.aten.convolution_backward,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    inputs = [
        grad_output,
        input_,
        weight,
        None,
        common_config[3],  # stride
        common_config[4],  # padding
        common_config[5],  # dilation
        True,
        common_config[6],
        common_config[7],
        [True, True, False],
    ]

    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32(inputs=inputs)
    else:
        test.check_result(inputs=inputs)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv_transpose3d_fwd_configs)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("channels_last", [True, False])
@pytest.mark.parametrize("dtype", get_conv_dtypes())
def test_conv_transpose3d_fwd(common_config, bias, channels_last, dtype):
    conv_args = {
        "in_channels": common_config[1],
        "out_channels": common_config[2],
        "kernel_size": common_config[3],
        "stride": common_config[4],
        "padding": common_config[5],
        "dilation": common_config[6],
        "output_padding": common_config[7],
        "groups": common_config[8],
        "bias": bias,
    }
    abs_diff, rel_diff = get_conv_test_tolerance("FWD")[dtype]
    memory_format = torch.channels_last_3d if channels_last else torch.contiguous_format

    func = torch.nn.ConvTranspose3d(**conv_args).to(memory_format=memory_format)
    test = testing.OpTest(
        func=func,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        test_dtype=dtype,
    )
    img = torch.randn(common_config[0]).type(dtype)
    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32({"input": img})
    else:
        test.check_result({"input": img})


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("common_config", conv_transpose3d_bwd_configs)
@pytest.mark.parametrize("dtype", get_conv_dtypes())
@pytest.mark.parametrize("channels_last", [True, False])
def test_conv_transpose3d_bwd(common_config, dtype, channels_last):
    weight_shape = [common_config[0][1], common_config[1][1], *common_config[2]]
    input_shape = common_config[0]
    grad_output_shape = common_config[1]
    memory_format = torch.channels_last_3d if channels_last else torch.contiguous_format

    weight = torch.randn(weight_shape, dtype=dtype).to(memory_format=memory_format)
    input_ = torch.randn(input_shape, dtype=dtype)
    grad_output = torch.randn(grad_output_shape, dtype=dtype)

    abs_diff, rel_diff = get_conv_test_tolerance("BWD")[dtype]
    test = testing.OpTest(
        func=torch.ops.aten.convolution_backward,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    inputs = [
        grad_output,
        input_,
        weight,
        None,
        common_config[3],  # stride
        common_config[4],  # padding
        common_config[5],  # dilation
        True,
        common_config[6],
        common_config[7],
        [True, True, False],
    ]

    if dtype == torch.float16:
        test.check_musafp16_vs_cpufp32(inputs=inputs)
    else:
        test.check_result(inputs=inputs)
