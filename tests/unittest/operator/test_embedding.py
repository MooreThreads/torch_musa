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
        (1, 128),
        (1, 512),
        (2, 16),
        (21, 128),
        (32, 128),
        (128, 128),
        (2, 256),
        (8, 256),
        (16, 256),
        (128, 256),
        (2, 512),
        (8, 512),
        (16, 512),
        (128, 512),
    ]


def gen_shape_of_indices_for_bwd():
    return [
        (512,),
        (1, 128),
        (1, 512),
        (2, 16),
        (21, 128),
        (32, 128),
        (2, 256),
        (8, 256),
        (16, 256),
        (2, 512),
        (8, 512),
    ]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", gen_shape_of_indices())
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_embedding(input_shape, weight_dtype, indices_dtype):
    global m, n
    input_tensor = torch.randint(low=0, high=n, size=input_shape).type(indices_dtype)
    embedding_args = {"num_embeddings": n, "embedding_dim": m}
    test = testing.OpTest(
        func=torch.nn.Embedding,
        input_args=embedding_args,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
        test_dtype=weight_dtype,
    )
    if weight_dtype == torch.float32:
        test.check_result(
            {"input": input_tensor}, train=True, dtype_nocast_map={"input": True}
        )
    else:
        test.check_musafp16_vs_musafp32(
            {"input": input_tensor}, train=True, dtype_nocast_map={"input": True}
        )


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
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

    # forward check #
    cpu_fwd_rst = fwd_func("cpu").type(torch.float32)
    musa_fwd_rst = fwd_func("musa").type(torch.float32)
    assert comparator(cpu_fwd_rst, musa_fwd_rst.cpu())


# backward check
n = random.randint(2048, 4096)
float_dtypes = [torch.float32, torch.float16]
# bf16 is not supported on arch older than qy2
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", gen_shape_of_indices_for_bwd())
@pytest.mark.parametrize("dtype", float_dtypes)
def test_embedding_bwd(input_shape, dtype):
    global m, n
    comparator = testing.DefaultComparator(abs_diff=1e-6)
    if dtype == torch.float16:
        comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=1e-3)
    if dtype == torch.bfloat16:
        # In contrast to MUSA's implementation, there is no accumulation of
        # intermediate computation in CPU's implementation, so relax the rel error.
        comparator = testing.DefaultComparator(abs_diff=6e-2, rel_diff=1e-2)
    grad_output = torch.randn((*input_shape, m), dtype=dtype)
    indices = torch.randint(low=0, high=n, size=input_shape)

    input_args = {
        "grad_output": grad_output,
        "indices": indices,
        "num_weights": n,
        "padding_idx": -1,
        "scale_grad_by_freq": False,
    }
    test = testing.OpTest(
        func=torch.ops.aten.embedding_dense_backward,
        input_args=input_args,
        comparators=comparator,
    )
    test.check_result()
