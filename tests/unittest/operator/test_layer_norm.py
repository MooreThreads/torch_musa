"""Test layer_norm operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,invalid-name,not-callable
import torch
import pytest
import torch_musa

from torch_musa import testing

data = [1, 2, 4, 8, 16]

@pytest.mark.parametrize("embedding_dim", data)
@pytest.mark.parametrize("batch", data)
@pytest.mark.parametrize("sequence_length", data)
def test_layer_norm_nlp(embedding_dim, batch, sequence_length):
    layer_norm = torch.nn.LayerNorm(embedding_dim)
    input_data = torch.randn(batch, sequence_length, embedding_dim)
    output = layer_norm(input_data)
    output_musa = layer_norm.to('musa')(input_data.to('musa'))
    assert testing.DefaultComparator(abs_diff=1e-3)(output, output_musa.cpu())


@pytest.mark.parametrize("N", data)
@pytest.mark.parametrize("C", data)
@pytest.mark.parametrize("W", data)
@pytest.mark.parametrize("H", data)
def test_layer_norm_cv(N, C, W, H):
    input_data = torch.randn(N, C, H, W)
    layer_norm = torch.nn.LayerNorm([C, H, W])
    output = layer_norm(input_data)
    output_musa = layer_norm.to('musa')(input_data.to('musa'))
    assert testing.DefaultComparator(abs_diff=1e-3)(output, output_musa.cpu())
