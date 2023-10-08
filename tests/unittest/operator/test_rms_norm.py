"""Test rms_norm operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,invalid-name,not-callable
import torch
import torch.nn.functional as F
import pytest
import torch_musa

from torch_musa import testing


def ref_rms_norm(input, normalized_shape, weight, eps):  # pylint: disable=W0622
    # Reshape through normalized_shape
    last_size = 1
    for xdim in normalized_shape:
        last_size *= xdim
    input_shape = input.shape
    input = input.reshape(input_shape[: -len(normalized_shape)] + (last_size,))
    weight = weight.view(-1)
    # Compute
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = input / (torch.sqrt(variance + eps))
    if weight.dtype in [torch.float16, torch.bfloat16]:
        hidden_states = hidden_states.to(weight.dtype)
    output = weight * hidden_states
    output = output.reshape(input_shape)
    return output


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("embedding_dim", [128, 512, 768, 2048])
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("sequence_length", [1, 32, 128])
def test_rms_norm_nlp(embedding_dim, batch, sequence_length):
    input_shape = (batch, sequence_length, embedding_dim)
    normalized_shape = (embedding_dim,)
    input_data = {
        "input": torch.randn(input_shape, dtype=torch.float16).to(torch.float32),
        "normalized_shape": normalized_shape,
        "weight": torch.randn(normalized_shape, dtype=torch.float16).to(torch.float32),
        "eps": 1e-8,
    }
    test = testing.OpTest(
        func=torch.rms_norm,
        refer_func=ref_rms_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test._comparators = [testing.DefaultComparator(abs_diff=1e-2)]
    test.check_musafp16_vs_musafp32()


cv_test_data = [2, 4, 8, 16]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("N", cv_test_data)
@pytest.mark.parametrize("C", cv_test_data)
@pytest.mark.parametrize("W", cv_test_data)
@pytest.mark.parametrize("H", cv_test_data)
def test_rms_norm_cv(N, C, W, H):
    normalized_shape = (C, H, W)
    input_data = {
        "input": torch.randn((N, C, H, W), dtype=torch.float16).to(torch.float32),
        "normalized_shape": normalized_shape,
        "weight": torch.randn(normalized_shape, dtype=torch.float16).to(torch.float32),
        "eps": 1e-8,
    }
    test = testing.OpTest(
        func=torch.rms_norm,
        refer_func=ref_rms_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test._comparators = [testing.DefaultComparator(abs_diff=1e-2)]
    test.check_musafp16_vs_musafp32()
