"""Test layer_norm operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,invalid-name,not-callable
import torch
import pytest
import torch_musa

from torch_musa import testing

data = [1, 2, 4, 8, 16]
dtypes = [torch.float32, torch.float16]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("embedding_dim", data)
@pytest.mark.parametrize("batch", data)
@pytest.mark.parametrize("sequence_length", data)
@pytest.mark.parametrize("dtype", dtypes)
def test_layer_norm_nlp(embedding_dim, batch, sequence_length, dtype):
    layer_norm = torch.nn.LayerNorm(embedding_dim)
    input_data = torch.randn(batch, sequence_length, embedding_dim)
    input_data.to(dtype)
    output = layer_norm(input_data)
    output_musa = layer_norm.to('musa')(input_data.to('musa'))
    assert testing.DefaultComparator(abs_diff=1e-3)(output, output_musa.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("N", data)
@pytest.mark.parametrize("C", data)
@pytest.mark.parametrize("W", data)
@pytest.mark.parametrize("H", data)
@pytest.mark.parametrize("dtype", dtypes)
def test_layer_norm_cv(N, C, W, H, dtype):
    input_data = torch.randn(N, C, H, W)
    layer_norm = torch.nn.LayerNorm([C, H, W])
    input_data.to(dtype)
    output = layer_norm(input_data)
    output_musa = layer_norm.to('musa')(input_data.to('musa'))
    assert testing.DefaultComparator(abs_diff=1e-3)(output, output_musa.cpu())
