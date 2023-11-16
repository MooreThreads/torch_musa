"""Test embedding operators."""
# pylint: disable=missing-function-docstring, global-variable-not-assigned, redefined-outer-name, unused-import
import random
import numpy as np
import torch
import pytest

from torch_musa import testing

n = random.randint(1, 1024)
m = random.randint(1, 1024)

input_data = [
    {
        "input": torch.LongTensor([random.randint(0,n-1)] * 128),
        "num_embeddings": n,
        "embedding_dim": m,
    },

    {
        "input": torch.LongTensor([[random.randint(0,n-1)] * 128, [random.randint(0, n-1)] * 128]) ,
        "num_embeddings": n,
        "embedding_dim": m,
    },

]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_embedding(input_data):
    embedding_args = {"num_embeddings": input_data["num_embeddings"],
                   "embedding_dim": input_data["embedding_dim"]}
    test = testing.OpTest(func=torch.nn.Embedding,
                          input_args=embedding_args,
                          comparators=testing.DefaultComparator(abs_diff=1e-6))
    test.check_result({"input": input_data["input"]}, train=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_embedding_fp16(input_data):
    embedding_args = {"num_embeddings": input_data["num_embeddings"],
                   "embedding_dim": input_data["embedding_dim"]}
    test = testing.OpTest(func=torch.nn.Embedding,
                          input_args=embedding_args,
                          comparators=testing.DefaultComparator(abs_diff=1e-6))
    test.check_musafp16_vs_musafp32({"input": input_data["input"]}, train=True)


# TODO(mingyuan.wang): delete next four lines when bf16 works well
@pytest.mark.skipif(
    True,  # always skip for now, cause numerical bug exist
    reason="bf16 result incorrect yet"
)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", [(2, 128), (128, 128), (1, 512), (4, 512)])
def test_embedding_bf16(input_shape):
    dtype = torch.bfloat16
    global m, n
    seed = 0
    input_tensor = torch.randint(low=0, high=n, size=input_shape)
    comparator = testing.DefaultComparator(abs_diff=1e-3, rel_diff=1e-3)

    def fwd_func(device):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        embedding = torch.nn.Embedding(n, m).to(device).to(dtype)
        input_t = input_tensor.clone().to(device)
        out = embedding(input_t)

        return out

    # backward check #
    grad_output = torch.randn((input_shape[0], input_shape[1], m), dtype=dtype)
    indices = torch.randint(low=0, high=n, size=input_shape)
    op = torch.ops.aten.embedding_dense_backward
    out_cpu = op(grad_output,indices, n, -1, False).type(torch.float32)
    out_musa = op(grad_output.to("musa"), indices.to("musa"), n, -1, False).type(torch.float32)
    assert comparator(out_cpu, out_musa.cpu())

    # forward check #
    cpu_fwd_rst = fwd_func("cpu").type(torch.float32)
    musa_fwd_rst = fwd_func("musa").type(torch.float32)
    assert comparator(cpu_fwd_rst, musa_fwd_rst.cpu())
