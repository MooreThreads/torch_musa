"""Test RReLU with noise operators."""

# pylint: disable=missing-function-docstring, unused-import
import random
import pytest
import torch
from torch_musa import testing

configs = [
    # shape
    [(1024,)],
    [(4, 256)],
    [(4, 256, 2)],
    [(4, 256, 2, 2)],
    [(4, 1, 20, 20)],
    [(4, 20, 1, 1)],
    [(5, 64, 30, 2, 3)],
    [(7, 1, 2048, 2, 3, 4)],
    [(2, 4, 8, 16, 32, 2, 2)],
    [(2, 3, 4, 5, 6, 7, 8, 1)],
]

lower, upper = 0.1, 0.3


def get_input_tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, requires_grad=True)
    if input_tensor.dim() == 4 and random.random() < 0.5:
        input_tensor = input_tensor.to(memory_format=torch.channels_last)
    return input_tensor


def get_noise_tensor(shape, dtype):
    return torch.empty(shape, dtype=dtype)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
def test_rrelu_with_noise_fp32(config):
    shape = config[0]

    def rrelu_fp32(input_tensor, noise, training):
        return torch.ops.aten.rrelu_with_noise(
            input_tensor, noise, lower, upper, training=training
        )

    input_tensor = get_input_tensor(shape, torch.float32)
    noise = get_noise_tensor(shape, torch.float32)
    test = testing.OpTest(
        func=rrelu_fp32,
        input_args={
            "input_tensor": input_tensor,
            "noise": noise,
            #  "lower": lower,
            #  "upper": upper,
            "training": False,
        },
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()


#  test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
def test_rrelu_with_noise_fp16(config):
    shape = config[0]

    def rrelu_fp16(input_tensor, noise, training):
        return torch.ops.aten.rrelu_with_noise(
            input_tensor, noise, lower, upper, training=training
        )

    input_tensor = get_input_tensor(shape, torch.float16)
    noise = get_noise_tensor(shape, torch.float16)
    test = testing.OpTest(
        func=rrelu_fp16,
        input_args={
            "input_tensor": input_tensor,
            "noise": noise,
            #  "lower": lower,
            #  "upper": upper,
            "training": False,
        },
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_musafp16_vs_musafp32(train=True)


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
def test_rrelu_with_noise_bf16(config):
    shape = config[0]

    def rrelu_bf16(input_tensor, noise, training):
        return torch.ops.aten.rrelu_with_noise(
            input_tensor, noise, lower, upper, training=training
        )

    input_tensor = get_input_tensor(shape, torch.bfloat16)
    noise = get_noise_tensor(shape, torch.bfloat16)
    test = testing.OpTest(
        func=rrelu_bf16,
        input_args={
            "input_tensor": input_tensor,
            "noise": noise,
            # "lower": lower,
            # "upper": upper,
            "training": False,
        },
        comparators=testing.DefaultComparator(abs_diff=1e-2),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
def test_rrelu_with_noise_inplace_fp32(config):
    shape = config[0]

    def rrelu_inplace(input_tensor, noise, training):
        return torch.ops.aten.rrelu_with_noise_(
            input_tensor, noise, lower, upper, training=training
        )

    input_tensor = get_input_tensor(shape, torch.float32)
    noise = get_noise_tensor(shape, torch.float32)
    test = testing.OpTest(
        func=rrelu_inplace,
        input_args={
            "input_tensor": input_tensor,
            "noise": noise,
            # "lower": lower,
            # "upper": upper,
            "training": False,
        },
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
def test_rrelu_with_noise_out_fp32(config):
    shape = config[0]

    def rrelu_out(input_tensor, noise, training, out):
        return torch.ops.aten.rrelu_with_noise(
            input_tensor, noise, lower, upper, training=training, out=out
        )

    input_tensor = get_input_tensor(shape, torch.float32)
    noise = get_noise_tensor(shape, torch.float32)
    output = torch.empty_like(input_tensor)
    test = testing.OpTest(
        func=rrelu_out,
        input_args={
            "input_tensor": input_tensor,
            "noise": noise,
            # "lower": lower,
            # "upper": upper,
            "training": False,
            "out": output,
        },
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
