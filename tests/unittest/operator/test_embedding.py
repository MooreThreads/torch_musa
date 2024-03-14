"""Test embedding operators."""
# pylint: disable=missing-function-docstring, global-variable-not-assigned, redefined-outer-name, unused-import
import random
import numpy as np
import torch
import pytest

from torch_musa import testing

n = random.randint(1, 1024)
m = random.randint(1, 1024)

def gen_shape_of_indices():
    return [
        (512,),
        (1, 128), (1, 512), (2, 16),
        (21, 128), (32, 128), (128, 128),
        (2, 256), (8, 256), (16, 256), (128, 256),
        (2, 512), (8, 512), (16, 512), (128, 512)
        ]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", gen_shape_of_indices())
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_embedding(input_shape, weight_dtype, indices_dtype):
    global m, n
    input_tensor = torch.randint(low=0, high=n, size=input_shape).type(indices_dtype)
    embedding_args = {"num_embeddings": n,
                      "embedding_dim": m}
    test = testing.OpTest(func=torch.nn.Embedding,
                          input_args=embedding_args,
                          comparators=testing.DefaultComparator(abs_diff=1e-6))
    if weight_dtype == torch.float32:
        test.check_result({"input": input_tensor}, train=True)
    else:
        test.check_musafp16_vs_musafp32({"input": input_tensor}, train=True)


@pytest.mark.skipif(
    True,  # always skip for now, bf16 dtype has low percision
    reason="bf16 has low percision"
)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="bf16 is not supported on arch older than S4000"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", gen_shape_of_indices())
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_embedding_bf16(input_shape, indices_dtype):
    dtype = torch.bfloat16
    global m, n
    seed = 0
    input_tensor = torch.randint(low=0, high=n, size=input_shape).type(indices_dtype)
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3)

    def fwd_func(device):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        embedding = torch.nn.Embedding(n, m).to(device).to(dtype)
        input_t = input_tensor.clone().to(device)
        out = embedding(input_t)

        return out

    # backward check #
    grad_output = torch.randn((*input_shape, m), dtype=dtype)
    indices = torch.randint(low=0, high=n, size=input_shape)
    op = torch.ops.aten.embedding_dense_backward
    out_cpu = op(grad_output,indices, n, -1, False).type(torch.float32)
    out_musa = op(grad_output.to("musa"), indices.to("musa"), n, -1, False).type(torch.float32)
    assert comparator(out_cpu, out_musa.cpu())

    # forward check #
    cpu_fwd_rst = fwd_func("cpu").type(torch.float32)
    musa_fwd_rst = fwd_func("musa").type(torch.float32)
    assert comparator(cpu_fwd_rst, musa_fwd_rst.cpu())
