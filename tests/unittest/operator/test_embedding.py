"""Test embedding operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random
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
    test.check_musafp16_vs_musafp32({"input": input_data["input"]}, train=False)
