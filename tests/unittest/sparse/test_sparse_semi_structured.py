"""Unit tests for semi-structured sparse ATen ops dispatched to MUSA.

These mirror the scenarios in PyTorch ``test/test_sparse_semi_structured.py`` but
use the ``musa`` device and CPU float references where a dense baseline exists.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random

import pytest
import torch
from torch.testing import make_tensor
from torch.sparse import SparseSemiStructuredTensorCUTLASS, to_sparse_semi_structured
from torch.sparse._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
)

import torch_musa
from torch_musa import testing

DEVICE = "musa"

atol_rtol_kw = {
    torch.float16: {"rtol": 5e-3, "atol": 5e-3},
    torch.bfloat16: {"rtol": 5e-2, "atol": 5e-2},
}


def rand_sparse_semi_structured(r, c, dtype, device, choice=None):
    """Build a 2:4 (or float32 1:2) structured sparse matrix on ``device``."""
    pattern = "2by4" if dtype != torch.float32 else "1by2"
    if pattern == "1by2":
        ksparse = 2
        choices = [[0, 1], [1, 0]]
    else:
        ksparse = 4
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    mask_entries = [choice or random.choice(choices) for _ in range(r * c // ksparse)]
    mask = torch.tensor(mask_entries, dtype=torch.bool).view(r, c).to(device)
    dense = make_tensor((r, c), dtype=dtype, device=device)
    dense[dense == 0] = 1
    dense = dense.masked_fill(~mask, 0)
    return dense


def effective_weight_from_tile(
    packed: torch.Tensor, meta: torch.Tensor, k: int, dtype: torch.dtype
) -> torch.Tensor:
    eye = torch.eye(k, dtype=dtype, device=packed.device)
    return torch._sparse_semi_structured_mm(packed, meta, eye)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_to_sparse_semi_structured_matches_cpu(dtype):
    m, k = 64, 128
    dense_cpu = rand_sparse_semi_structured(m, k, dtype, "cpu")
    dense = dense_cpu.to(DEVICE)
    packed_ref, meta_ref = sparse_semi_structured_from_dense_cutlass(dense_cpu)
    packed, meta = torch._to_sparse_semi_structured(dense)
    torch.testing.assert_close(packed.cpu(), packed_ref, rtol=0, atol=0)
    torch.testing.assert_close(meta.cpu(), meta_ref, rtol=0, atol=0)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_semi_structured_tile(dtype):
    """Packed and packed_t should contain the same multiset of values (PyTorch heuristic test)."""
    n = 512
    torch.manual_seed(0)
    a = torch.randn(n, n, dtype=dtype, device=DEVICE)
    x = torch.eye(n, dtype=dtype, device=DEVICE)
    packed, meta, packed_t, meta_t = torch._sparse_semi_structured_tile(a)[:4]
    torch.testing.assert_close(
        packed.to(torch.float64).sum(),
        packed_t.to(torch.float64).sum(),
    )
    a_eff = effective_weight_from_tile(packed, meta, n, dtype)
    ref_linear = x @ a_eff.t()
    out_linear = torch._sparse_semi_structured_linear(x, packed, meta)
    max_diff = (ref_linear - out_linear).abs().argmax()
    torch.testing.assert_close(
        ref_linear,
        out_linear,
        **atol_rtol_kw[dtype],
        msg=f"packed linear mismatch at flat index {max_diff.item()}",
    )
    a_eff_t = effective_weight_from_tile(packed_t, meta_t, n, dtype)
    ref_linear_t = x @ a_eff_t.t()
    out_linear_t = torch._sparse_semi_structured_linear(x, packed_t, meta_t)
    max_diff = (ref_linear_t - out_linear_t).abs().argmax()
    torch.testing.assert_close(
        ref_linear_t,
        out_linear_t,
        **atol_rtol_kw[dtype],
        msg=f"packed_t linear mismatch at flat index {max_diff.item()}",
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_semi_structured_tile_edge_quad(dtype):
    quad = torch.tensor(
        [
            [2, -1, -2, -3],
            [-1, 8, -1, 6],
            [-1, -1, 4, 5],
            [-1, 3, 7, -1],
        ],
        dtype=dtype,
        device=DEVICE,
    )
    a = torch.randn(32, 64, dtype=dtype, device=DEVICE)
    a[:4, :4] = quad
    packed, meta, packed_t, _ = torch._sparse_semi_structured_tile(a)[:4]
    assert packed[0, 0].item() == 2
    assert packed[0, 1].item() == 0
    assert packed_t[0, 0].item() == 2
    assert packed_t[0, 1].item() == 0
    assert meta.shape[0] == packed.shape[0]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_semi_structured_apply(dtype):
    m, n = 256, 1024
    x = torch.randn(m, n, dtype=dtype, device=DEVICE)
    packed, meta, packed_t, meta_t, bitmask = torch._sparse_semi_structured_tile(x)
    packed2, packed_t2 = torch._sparse_semi_structured_apply(x, bitmask)
    torch.testing.assert_close(packed, packed2)
    torch.testing.assert_close(packed_t, packed_t2)
    assert meta.numel() > 0 and meta_t.numel() > 0


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_semi_structured_apply_dense(dtype):
    m, n = 256, 1024  # alignment-friendly

    x = torch.randn(m, n, dtype=dtype, device=DEVICE)

    bitmask = torch._sparse_semi_structured_tile(x)[4]

    packed_ref, packed_t_ref = torch._sparse_semi_structured_apply(x, bitmask)

    dense = torch._sparse_semi_structured_apply_dense(x, bitmask)

    packed_from_dense, packed_t_from_dense = torch._sparse_semi_structured_apply(
        dense, bitmask
    )

    torch.testing.assert_close(
        packed_from_dense,
        packed_ref,
        **atol_rtol_kw[dtype],
        msg="apply_dense -> apply packed mismatch",
    )
    torch.testing.assert_close(
        packed_t_from_dense,
        packed_t_ref,
        **atol_rtol_kw[dtype],
        msg="apply_dense -> apply packed_t mismatch",
    )

    # 基本 sanity
    assert dense.shape == x.shape
    assert dense.dtype == x.dtype


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_semi_structured_mm(dtype):
    m, n, k = 256, 384, 512
    a = torch.randn(m, k, dtype=dtype, device=DEVICE)
    b = torch.randn(k, n, dtype=dtype, device=DEVICE)
    packed, meta, packed_t, meta_t = torch._sparse_semi_structured_tile(a)[:4]
    out_mm = torch._sparse_semi_structured_mm(packed, meta, b)
    a_eff = effective_weight_from_tile(packed, meta, k, dtype)
    ref_mm = a_eff @ b
    max_diff = (ref_mm - out_mm).abs().argmax()
    torch.testing.assert_close(
        ref_mm,
        out_mm,
        **atol_rtol_kw[dtype],
        msg=f"packed mm mismatch at flat index {max_diff.item()}",
    )

    b_t = torch.randn(m, n, dtype=dtype, device=DEVICE)
    out_mm_t = torch._sparse_semi_structured_mm(packed_t, meta_t, b_t)
    a_eff_t = effective_weight_from_tile(packed_t, meta_t, m, dtype)
    ref_mm_t = a_eff_t @ b_t
    max_diff = (ref_mm_t - out_mm_t).abs().argmax()
    torch.testing.assert_close(
        ref_mm_t,
        out_mm_t,
        **atol_rtol_kw[dtype],
        msg=f"packed_t mm mismatch at flat index {max_diff.item()}",
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_semi_structured_addmm(dtype):
    # m, n, k = 64, 256, 1024
    m, n, k = 256, 384, 512
    inp = make_tensor((m,), dtype=dtype, device=DEVICE)
    alpha = 1.3
    beta = -0.7
    a = torch.randn(m, k, dtype=dtype, device=DEVICE)
    b = torch.randn(k, n, dtype=dtype, device=DEVICE)
    packed, meta, packed_t, meta_t = torch._sparse_semi_structured_tile(a)[:4]
    out_mm = torch._sparse_semi_structured_addmm(
        inp, packed, meta, b, alpha=alpha, beta=beta
    )
    a_eff = effective_weight_from_tile(packed, meta, k, dtype)
    ref_mm = beta * inp.unsqueeze(1) + alpha * (a_eff @ b)
    max_diff = (ref_mm - out_mm).abs().argmax()
    torch.testing.assert_close(
        ref_mm,
        out_mm,
        **atol_rtol_kw[dtype],
        msg=f"packed mm mismatch at flat index {max_diff.item()}",
    )

    b_t = torch.randn(m, n, dtype=dtype, device=DEVICE)
    inp_t = make_tensor((k,), dtype=dtype, device=DEVICE)
    out_t = torch._sparse_semi_structured_addmm(
        inp_t, packed_t, meta_t, b_t, alpha=alpha, beta=beta
    )
    a_eff_t = effective_weight_from_tile(packed_t, meta_t, m, dtype)
    ref_t = beta * inp_t.unsqueeze(1) + alpha * (a_eff_t @ b_t)
    max_diff = (ref_t - out_t).abs().argmax()
    torch.testing.assert_close(
        ref_t,
        out_t,
        **atol_rtol_kw[dtype],
        msg=f"packed_t mm mismatch at flat index {max_diff.item()}",
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_shape", [[], [3], [3, 1]])
@pytest.mark.parametrize("add_bias", [False, True])
def test_sparse_semi_structured_linear(dtype, batch_shape, add_bias):
    m, n, k = 32, 32, 128
    weight = rand_sparse_semi_structured(m, k, dtype, DEVICE)
    input_t = make_tensor((*batch_shape, n, k), dtype=dtype, device=DEVICE)
    bias = make_tensor((m,), dtype=dtype, device=DEVICE) if add_bias else None

    dtype_dense = torch.float32
    input_dense = input_t.to(dtype_dense)
    weight_dense = weight.to(dtype_dense)
    bias_dense = bias.to(dtype_dense) if add_bias else None
    output_ref = torch.nn.functional.linear(input_dense, weight_dense, bias_dense)

    weight_sparse, meta = torch._to_sparse_semi_structured(weight)
    output = torch._sparse_semi_structured_linear(
        input_t, weight_sparse, meta, bias=bias
    )
    torch.testing.assert_close(
        output.to(dtype_dense), output_ref, **atol_rtol_kw[dtype]
    )
