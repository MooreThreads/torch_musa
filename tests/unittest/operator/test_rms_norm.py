"""Test rms_norm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,invalid-name,not-callable
import torch
import torch.nn.functional as F
import pytest
import torch_musa

from torch_musa import testing


def ref_rms_norm(input, normalized_shape, weight, eps):  # pylint: disable=W0622
    # to avoid underflow / overflow
    input = input.to(torch.float32)
    weight = weight.to(torch.float32)
    last_size = 1
    for xdim in normalized_shape:
        last_size *= xdim
    input_shape = input.shape
    input = input.reshape(input_shape[: -len(normalized_shape)] + (last_size,))
    weight = weight.view(-1)
    # Compute
    variance = input.pow(2).mean(-1, keepdim=True)
    hidden_states = input * torch.rsqrt(variance + eps)
    output = weight * hidden_states
    output = output.reshape(input_shape)
    return output  # just return float32 output


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("embedding_dim", [128, 512, 768, 2048])
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("sequence_length", [1, 32, 128])
@pytest.mark.parametrize("dtype", [torch.half, torch.float32])
def test_rms_norm_nlp(embedding_dim, batch, sequence_length, dtype):
    input_shape = (batch, sequence_length, embedding_dim)
    normalized_shape = (embedding_dim,)
    input_data = {
        "input": torch.randn(input_shape, dtype=dtype),
        "normalized_shape": normalized_shape,
        "weight": torch.randn(normalized_shape, dtype=dtype),
        "eps": 1e-8,
    }
    if dtype == torch.half:
        atol, rtol = 1e-3, 1e-2
    elif dtype == torch.float32:
        atol, rtol = 1e-6, 1e-6
    test = testing.OpTest(
        func=torch.rms_norm,
        refer_func=ref_rms_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(rel_diff=rtol, abs_diff=atol),
    )
    if dtype == torch.half:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.float32:
        test.check_result()


cv_test_data = [2, 4, 8, 16]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("N", cv_test_data)
@pytest.mark.parametrize("C", cv_test_data)
@pytest.mark.parametrize("W", cv_test_data)
@pytest.mark.parametrize("H", cv_test_data)
@pytest.mark.parametrize("dtype", [torch.half, torch.float32])
def test_rms_norm_cv(N, C, W, H, dtype):
    normalized_shape = (C, H, W)
    input_data = {
        "input": torch.randn((N, C, H, W), dtype=dtype),
        "normalized_shape": normalized_shape,
        "weight": torch.randn(normalized_shape, dtype=dtype),
        "eps": 1e-8,
    }
    if dtype == torch.half:
        atol, rtol = 1e-3, 1e-2
    elif dtype == torch.float32:
        atol, rtol = 1e-6, 1e-6
    test = testing.OpTest(
        func=torch.rms_norm,
        refer_func=ref_rms_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(rel_diff=rtol, abs_diff=atol),
    )
    if dtype == torch.half:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.float32:
        test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("embedding_dim", [128, 512, 768, 2048])
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("sequence_length", [1, 32, 128])
@pytest.mark.parametrize("dtype", [torch.half, torch.float32])
def test_rms_norm_backward(embedding_dim, batch, sequence_length, dtype):
    input_shape = (batch, sequence_length, embedding_dim)
    normalized_shape = (embedding_dim,)
    input_data = {
        "input": torch.randn(input_shape, dtype=dtype, requires_grad=True),
        "normalized_shape": normalized_shape,
        "weight": torch.randn(normalized_shape, dtype=dtype, requires_grad=True),
        "eps": 1e-6,
    }
    if dtype == torch.half:
        atol, rtol = 5e-2, 1e-3
    elif dtype == torch.float32:
        atol, rtol = 1e-4, 1e-5  # grad has bigger float pointing error.
    test = testing.OpTest(
        func=torch.rms_norm,
        refer_func=ref_rms_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(rel_diff=rtol, abs_diff=atol),
    )
    if dtype == torch.half:
        test.check_musafp16_vs_musafp32(train=True)
    elif dtype == torch.float32:
        test.check_result(train=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("embedding_dim", [128, 512, 768])
@pytest.mark.parametrize("stride_sequence_length", [2112])
@pytest.mark.parametrize("sequence_length", [7, 32])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float32])
def test_rms_norm_2d_noncontiguous(
    embedding_dim, stride_sequence_length, sequence_length, dtype
):
    input_shape = (sequence_length, embedding_dim)
    normalized_shape = (embedding_dim,)

    storage_size = sequence_length * stride_sequence_length
    base_tensor = torch.randn(storage_size).musa()

    input_tensor = base_tensor.as_strided(
        size=input_shape, stride=(stride_sequence_length, 1), storage_offset=0
    )

    input_data = {
        "input": input_tensor,
        "normalized_shape": normalized_shape,
        "weight": torch.randn(normalized_shape, dtype=dtype, requires_grad=True),
        "eps": 1e-6,
    }
    if dtype in (torch.half, torch.bfloat16):
        atol, rtol = 5e-2, 1e-3
    elif dtype == torch.float32:
        atol, rtol = 1e-4, 1e-5  # grad has bigger float pointing error.
    test = testing.OpTest(
        func=torch.rms_norm,
        refer_func=ref_rms_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(rel_diff=rtol, abs_diff=atol),
    )
    if dtype == torch.half:
        test.check_musafp16_vs_musafp32()
    if dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
    elif dtype == torch.float32:
        test.check_result()
