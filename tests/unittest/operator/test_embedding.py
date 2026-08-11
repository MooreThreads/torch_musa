"""Test embedding operators."""

# pylint: disable=missing-function-docstring, global-variable-not-assigned, redefined-outer-name, unused-import
import random
import numpy as np
import torch
import pytest

from torch_musa import testing

n = random.randint(513, 1024)
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


def get_embedding_bwd_comparator(dtype):
    if dtype == torch.float16:
        return testing.DefaultComparator(abs_diff=5e-2, rel_diff=1e-3)
    if dtype == torch.bfloat16:
        return testing.DefaultComparator(abs_diff=6e-2, rel_diff=1e-2)
    return testing.DefaultComparator(abs_diff=1e-6)


def gen_scale_grad_by_freq_indices_cases():
    return [
        (
            "partial_duplicate",
            torch.tensor(
                [
                    [4, 6, 5, 6],
                    [9, 6, 5, 5],
                ],
                dtype=torch.long,
            ),
        ),
        (
            "all_same",
            torch.zeros((8, 512), dtype=torch.long),
        ),
        (
            "all_unique",
            torch.arange(8 * 512, dtype=torch.long).reshape(8, 512),
        ),
    ]


def gen_embedding_bwd_cases():
    cases = [
        pytest.param(
            False,
            input_shape,
            None,
            id=f"dense_backward_{input_shape}",
        )
        for input_shape in gen_shape_of_indices_for_bwd()
    ]
    cases.extend(
        pytest.param(
            True,
            None,
            indices,
            id=f"scale_grad_by_freq_{case_id}",
        )
        for case_id, indices in gen_scale_grad_by_freq_indices_cases()
    )
    return cases


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize(
    "scale_grad_by_freq,input_shape,fixed_indices",
    gen_embedding_bwd_cases(),
)
def test_embedding_bwd(dtype, scale_grad_by_freq, input_shape, fixed_indices):
    global m, n
    comparator = get_embedding_bwd_comparator(dtype)

    if fixed_indices is None:
        embedding_dim = m
        num_weights = n
        indices = torch.randint(low=0, high=num_weights, size=input_shape)
    else:
        embedding_dim = 128
        indices = fixed_indices
        num_weights = int(indices.max()) + 1

    grad = torch.randn((*indices.shape, embedding_dim), dtype=dtype)

    input_args = {
        "grad": grad,
        "indices": indices,
        "num_weights": num_weights,
        "padding_idx": -1,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": False,
    }
    test = testing.OpTest(
        func=torch.ops.aten.embedding_backward,
        input_args=input_args,
        comparators=comparator,
    )
    test.check_result()

    if scale_grad_by_freq:
        unscaled_input_args = {
            **input_args,
            "scale_grad_by_freq": False,
        }
        scaled_result = torch.ops.aten.embedding_backward(**input_args)
        unscaled_result = torch.ops.aten.embedding_backward(**unscaled_input_args)

        if (
            torch.count_nonzero(torch.bincount(indices.reshape(-1))).item()
            < indices.numel()
        ):
            assert not comparator(scaled_result.float(), unscaled_result.float())


float_dtypes = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", [(2, 32)])
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("norm_type", [1.0, 2.0])
@pytest.mark.skipif(testing.get_musa_arch() < 31, reason="Precision issue on QY2")
def test_embedding_renorm(input_shape, dtype, norm_type):
    comparator = testing.DefaultComparator(abs_diff=1e-3, rel_diff=1e-3)
    if dtype == torch.float16:
        comparator = testing.DefaultComparator(abs_diff=1e-3, rel_diff=2e-2)
    if dtype == torch.bfloat16:
        comparator = testing.DefaultComparator(abs_diff=1e-3, rel_diff=2e-2)

    n = random.randint(128 + 1, 256)
    m = 128
    input_tensor = torch.rand((n + 1, m)).to(dtype)
    indices = torch.randint(low=0, high=n, size=input_shape)
    max_norm = norm_type
    input_args = {
        "input": input_tensor,
        "indices": indices,
        "max_norm": max_norm,
        "norm_type": norm_type,
    }
    test = testing.OpTest(
        func=torch.embedding_renorm_,
        input_args=input_args,
        comparators=comparator,
        # test_dtype=dtype,
    )
    test.check_result()
