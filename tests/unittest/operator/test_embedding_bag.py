"""Test embedding_bag operators."""

# pylint: disable=missing-function-docstring, global-variable-not-assigned, redefined-outer-name, unused-import
import random
import torch
import pytest

from torch_musa import testing

n = random.randint(1, 1024)
m = random.randint(1, 1024)

input_data = [
    {
        "input": torch.randint(0, n, [128], dtype=torch.long),
        "num_embeddings": n,
        "embedding_dim": m,
    },
    {
        "input": torch.randint(0, n, [64], dtype=torch.long),
        "num_embeddings": n,
        "embedding_dim": m,
    },
    {
        "input": torch.randint(0, n, [32], dtype=torch.long),
        "num_embeddings": n,
        "embedding_dim": m,
    },
]
modes = ["sum", "mean", "max"]
sparses = [False]  # Musa does not support SparseTensor


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("mode", modes)
def test_embedding_bag_1d(input_data, mode):
    embedding_args = {
        "num_embeddings": input_data["num_embeddings"],
        "embedding_dim": input_data["embedding_dim"],
        "mode": mode,
    }
    offsets = torch.randint(0, input_data["input"].shape[0], [64])
    offsets = torch.sort(offsets)[0]
    offsets[0] = 0
    offsets[-1] = input_data["input"].shape[0]
    test = testing.OpTest(
        func=torch.nn.EmbeddingBag,
        input_args=embedding_args,
        comparators=testing.DefaultComparator(abs_diff=1e-6, rel_diff=1e-6),
        test_dtype=torch.float32,
    )
    test.check_result(
        {"input": input_data["input"], "offsets": offsets},
        train=False,
        dtype_nocast_map={"input": True, "offsets": True},
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("sparse", sparses)
def test_embedding_bag_backward(input_data, mode, sparse):
    if mode == "max":
        return
    atol = 1e-6
    rtol = 1e-6
    offsets = torch.randint(0, input_data["input"].shape[0], [64])
    offsets = torch.sort(offsets)[0]
    offsets[0] = 0
    offsets[-1] = input_data["input"].shape[0]
    embedding_args = {
        "num_embeddings": input_data["num_embeddings"],
        "embedding_dim": input_data["embedding_dim"],
        "mode": mode,
        "sparse": sparse,
    }
    test = testing.OpTest(
        func=torch.nn.EmbeddingBag,
        input_args=embedding_args,
        comparators=testing.DefaultComparator(rel_diff=rtol, abs_diff=atol),
        test_dtype=torch.float32,
    )
    test.check_result(
        {"input": input_data["input"], "offsets": offsets},
        train=True,
        dtype_nocast_map={"input": True, "offsets": True},
    )


input_data = [
    {
        "input": torch.randint(0, n, [16, 128], dtype=torch.long),
        "num_embeddings": n,
        "embedding_dim": m,
    },
    {
        "input": torch.randint(0, n, [8, 24], dtype=torch.long),
        "num_embeddings": n,
        "embedding_dim": m,
    },
    {
        "input": torch.randint(0, n, [3, 10], dtype=torch.long),
        "num_embeddings": n,
        "embedding_dim": m,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("sparse", sparses)
def test_embedding_bag_2d(input_data, mode, sparse):
    # max mode not support sparse
    if sparse and mode == "max":
        return
    embedding_args = {
        "num_embeddings": input_data["num_embeddings"],
        "embedding_dim": input_data["embedding_dim"],
        "mode": mode,
        "sparse": sparse,
    }
    test = testing.OpTest(
        func=torch.nn.EmbeddingBag,
        input_args=embedding_args,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
        test_dtype=torch.float32,
    )
    test.check_result(
        {"input": input_data["input"]}, train=False, dtype_nocast_map={"input": True}
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("mode", modes)
def notest_embedding_bag_fp16(input_data, mode):
    embedding_args = {
        "num_embeddings": input_data["num_embeddings"],
        "embedding_dim": input_data["embedding_dim"],
        "mode": mode,
    }
    test = testing.OpTest(
        func=torch.nn.EmbeddingBag,
        input_args=embedding_args,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_musafp16_vs_musafp32({"input": input_data["input"]}, train=True)
