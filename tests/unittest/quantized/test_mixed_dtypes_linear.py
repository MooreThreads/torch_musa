"""Tests for _mixed_dtypes_linear operator (W4A16 quantized GEMM)."""

# pylint: disable=missing-function-docstring, redefined-outer-name
import math

import pytest
import torch

from torch_musa import testing
from torch_musa.core.ao._quantized_conversions import (
    prepack_int4_weight_for_mixed_dtypes_linear,
    prepack_scale_for_mixed_dtypes_linear,
)


def _quantize_s4_per_group(weight, group_size):
    """Per-group symmetric S4 quantization on K axis for weight (K, N).

    Following moorcat's quantization scheme:
    - Quantizes to signed int4 range [-8, 7]
    - Returns quantized weight, per-group scales, and dequantized reference

    Returns:
        w_q: (K, N) int16, quantized weight values in [-8, 7]
        scales: (scale_k, N) float32, per-group scale factors
        w_ref: (K, N) float32, dequantized reference weight
    """
    k, n = weight.shape
    scale_k = (k + group_size - 1) // group_size

    w_q = torch.empty((k, n), dtype=torch.int16, device=weight.device)
    w_ref = torch.empty_like(weight, dtype=torch.float32)
    scales = torch.empty((scale_k, n), dtype=torch.float32, device=weight.device)

    for group_idx in range(scale_k):
        k_start = group_idx * group_size
        k_end = min((group_idx + 1) * group_size, k)
        w_slice = weight[k_start:k_end].to(torch.float32)

        scale = w_slice.abs().amax(dim=0).clamp(min=1e-6) / 7.0
        quantized = torch.round(w_slice / scale).clamp(-8, 7).to(torch.int16)

        w_q[k_start:k_end] = quantized
        scales[group_idx] = scale
        w_ref[k_start:k_end] = quantized.to(torch.float32) * scale

    return w_q, scales, w_ref


def _prepare_w4a16_data(m, k, n, group_size, dtype, device="musa"):
    """Prepare quantized W4A16 test data following moorcat's pattern.

    Returns:
        input_tensor: (M, K) fp16/bf16
        weight_packed: prepacked uint8 weight for the kernel
        scales_packed: (k_tiles, N) float32, prepacked scale
        w_ref: (K, N) float32, dequantized reference weight
    """
    torch.manual_seed(0)
    a = (3.0 * torch.rand((m, k), device=device) - 2.0).to(dtype)
    weight = (3.0 * torch.rand((k, n), device=device) - 1.0).to(dtype)

    w_q, scales, w_ref = _quantize_s4_per_group(weight, group_size=group_size)
    w_packed = prepack_int4_weight_for_mixed_dtypes_linear(w_q)
    scales_packed = prepack_scale_for_mixed_dtypes_linear(scales, k)

    return a, w_packed, scales_packed, w_ref


def _reference_mixed_dtypes_linear(input_tensor, w_ref, bias=None, activation=None):
    """CPU reference using pre-dequantized weight.

    Args:
        input_tensor: (*, K) fp16/bf16
        w_ref: (K, N) float32, dequantized weight
        bias: optional (N,) same dtype as input
        activation: None / "none" / "relu" / "silu"
    """
    dtype = input_tensor.dtype
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    out_features = w_ref.size(1)

    output = torch.matmul(input_2d.float(), w_ref).to(dtype)

    if bias is not None:
        output = output + bias.unsqueeze(0)

    if activation == "relu":
        output = torch.nn.functional.relu(output)
    elif activation == "silu":
        output = torch.nn.functional.silu(output)

    output_shape = list(input_tensor.shape)
    output_shape[-1] = out_features
    return output.reshape(output_shape)


# ============================================================================
# Test parameters
# ============================================================================
# [M, N, K] data format: n % 256 == 0, k % 128 == 0
_shapes_per_group = [
    (256, 256, 128),
    (128, 256, 512),
]

_dtypes = [torch.float16, torch.bfloat16]
_activations = ["none", "relu", "silu"]
GROUP_SIZE = 128  # group size must >= 128 & % 128 == 0& % 128 == 0


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 31,
    reason="_mixed_dtypes_linear requires MP31 or later",
)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("m, n, k", _shapes_per_group)
def test_mixed_dtypes_linear_per_group(dtype, m, n, k):
    """Per-group scale test following moorcat's quantization pattern."""

    a, w_packed, scales, w_ref = _prepare_w4a16_data(
        m, k, n, GROUP_SIZE, dtype, device="musa"
    )

    output = torch.ops.aten._mixed_dtypes_linear(a, w_packed, scales)

    assert output.dtype == dtype
    assert output.device.type == "musa"
    assert tuple(output.shape) == (m, n)

    ref = _reference_mixed_dtypes_linear(a.cpu(), w_ref.cpu())
    atol = min(5e-2 * math.sqrt(k), 1.0)
    rtol = 1e-1
    torch.testing.assert_close(output.cpu(), ref, rtol=rtol, atol=atol)


# ============================================================================
# Functional test — per-channel scale (group_size=K, original pytorch implementation)
# ============================================================================


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 31,
    reason="_mixed_dtypes_linear requires MP31 or later",
)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("m, n, k", _shapes_per_group)
def test_mixed_dtypes_linear_per_channel(dtype, m, n, k):
    """Per-channel scale test: group_size=K, prepack expands to (k_tiles, N)."""

    a, w_packed, scales, w_ref = _prepare_w4a16_data(
        m, k, n, group_size=k, dtype=dtype, device="musa"
    )

    output = torch.ops.aten._mixed_dtypes_linear(a, w_packed, scales)

    assert output.dtype == dtype
    assert tuple(output.shape) == (m, n)

    ref = _reference_mixed_dtypes_linear(a.cpu(), w_ref.cpu())
    atol = min(5e-2 * math.sqrt(k), 1.0)
    rtol = 1e-1
    torch.testing.assert_close(output.cpu(), ref, rtol=rtol, atol=atol)


# ============================================================================
# Functional tests — with bias and activation
# ============================================================================


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 31,
    reason="_mixed_dtypes_linear requires MP31 or later",
)
@pytest.mark.parametrize("dtype", _dtypes)
@pytest.mark.parametrize("activation", _activations)
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("m, n, k", _shapes_per_group)
def test_mixed_dtypes_linear_bias_activation(dtype, activation, add_bias, m, n, k):
    """Test bias + activation combinations."""
    a, w_packed, scales, w_ref = _prepare_w4a16_data(
        m, k, n, GROUP_SIZE, dtype, device="musa"
    )
    bias = torch.randn(n, dtype=dtype, device="musa") * 0.1 if add_bias else None

    output = torch.ops.aten._mixed_dtypes_linear(
        a,
        w_packed,
        scales,
        bias=bias,
        activation=activation,
    )

    assert output.dtype == dtype
    assert tuple(output.shape) == (m, n)

    ref = _reference_mixed_dtypes_linear(
        a.cpu(),
        w_ref.cpu(),
        bias=bias.cpu() if bias is not None else None,
        activation=activation,
    )
    atol = min(5e-2 * math.sqrt(k), 1.0)
    rtol = 1e-1
    torch.testing.assert_close(output.cpu(), ref, rtol=rtol, atol=atol)
