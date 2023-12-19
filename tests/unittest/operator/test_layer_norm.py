"""Test layer_norm operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,invalid-name,not-callable
import torch
import torch.nn.functional as F
import pytest
import torch_musa

from torch_musa import testing


def init_weight_and_bias(normalized_shape):
    weight = torch.randn(normalized_shape)
    bias = torch.randn(normalized_shape)
    return weight, bias


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("embedding_dim", [128, 512, 768, 2048])
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("sequence_length", [1, 32, 128])
def test_layer_norm_nlp(embedding_dim, batch, sequence_length):
    normalized_shape = (embedding_dim,)
    weight, bias = init_weight_and_bias(normalized_shape)
    input_data = {
        "input": torch.randn(batch, sequence_length, embedding_dim),
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": 1e-8,
    }
    test = testing.OpTest(
        func=F.layer_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-2),
    )
    test.check_result()
    test._comparators = [testing.DefaultComparator(abs_diff=1e-2)]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="fp16 layer norm supported in QY2 or later"
)
@pytest.mark.parametrize("embedding_dim", [128, 512, 768, 2048])
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("sequence_length", [1, 32, 128])
def test_layer_norm_nlp_fp16(embedding_dim, batch, sequence_length):
    normalized_shape = (embedding_dim,)
    weight, bias = init_weight_and_bias(normalized_shape)
    input_data = {
        "input": torch.randn(batch, sequence_length, embedding_dim),
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": 1e-8,
    }
    test = testing.OpTest(
        func=F.layer_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-2),
    )
    test._comparators = [testing.DefaultComparator(abs_diff=1e-2)]
    test.check_musafp16_vs_musafp32()


@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("embedding_dim", [128, 512, 768, 2048])
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("sequence_length", [1, 32, 128])
def test_layer_norm_nlp_bf16(embedding_dim, batch, sequence_length):
    normalized_shape = (embedding_dim,)
    weight, bias = init_weight_and_bias(normalized_shape)
    input_args = {
        "input": 
        torch.randn(
            (batch, sequence_length, embedding_dim),
            dtype=torch.bfloat16).requires_grad_(),
        "normalized_shape": normalized_shape,
        "weight": weight.to(torch.bfloat16),
        "bias": bias.to(torch.bfloat16),
        "eps": 1e-8,
    }
    test = testing.OpTest(
        func=F.layer_norm,
        input_args=input_args,
        comparators=testing.DefaultComparator(abs_diff=1e-2, rel_diff=1e-2),
    )
    test.check_result(train=True)


cv_test_data = [2, 4, 8, 16]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("N", cv_test_data)
@pytest.mark.parametrize("C", cv_test_data)
@pytest.mark.parametrize("W", cv_test_data)
@pytest.mark.parametrize("H", cv_test_data)
def test_layer_norm_cv(N, C, W, H):
    normalized_shape = [C, H, W]
    weight, bias = init_weight_and_bias(normalized_shape)
    input_data = {
        "input": torch.randn(N, C, H, W),
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": 1e-8,
    }
    test = testing.OpTest(
        func=F.layer_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test._comparators = [testing.DefaultComparator(abs_diff=1e-2)]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="fp16 layer norm supported in QY2 or later"
)
@pytest.mark.parametrize("N", cv_test_data)
@pytest.mark.parametrize("C", cv_test_data)
@pytest.mark.parametrize("W", cv_test_data)
@pytest.mark.parametrize("H", cv_test_data)
def test_layer_norm_cv_fp16(N, C, W, H):
    normalized_shape = [C, H, W]
    weight, bias = init_weight_and_bias(normalized_shape)
    input_data = {
        "input": torch.randn(N, C, H, W),
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": 1e-8,
    }
    test = testing.OpTest(
        func=F.layer_norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test._comparators = [testing.DefaultComparator(abs_diff=1e-2)]
    test.check_musafp16_vs_musafp32()


@pytest.mark.parametrize("dtype", [torch.float32])
def test_layer_norm_weight_uncontiguous(dtype):
    input_data = torch.tensor([1, 2, 3, 4, 5], dtype=dtype)
    layer = torch.nn.LayerNorm(5)
    cpu_res = layer(input_data)
    layer = layer.to("musa")
    chunk = torch.zeros(10).to("musa")
    chunk[5:].copy_(layer.weight)
    layer.weight.data = chunk[5:].view(layer.weight.shape)
    musa_res = layer(input_data.to("musa"))
    assert testing.DefaultComparator(abs_diff=1e-3)(cpu_res, musa_res.cpu())
