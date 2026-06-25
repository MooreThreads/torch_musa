"""Sparse COO tests for div.Tensor / div_.Tensor / div.out and *_mode variants.

PyTorch sparse COO division only allows divisor to be a scalar or a 0-dim dense
tensor (not sparse / not arbitrary-shaped dense).
"""

# pylint: disable=missing-function-docstring, C0116
import pytest
import torch
from utils import make_coo, coo_to_musa, assert_dense_close, MUSPARSE_LT_12000

dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]


def _scalar_tensor(value, *, dtype, device):
    """0-dim dense tensor on ``device`` (valid sparse divisor in PyTorch)."""
    return torch.tensor(value, dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", dtypes)
def test_div_tensor_sparse_coo(dtype):
    shape, nnz = (7, 9), 14
    a_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    a_musa = coo_to_musa(a_cpu)
    div_cpu = _scalar_tensor(2.5, dtype=dtype, device="cpu")
    div_musa = _scalar_tensor(2.5, dtype=dtype, device="musa")

    y_cpu = torch.ops.aten.div.Tensor(a_cpu, div_cpu)
    y_musa = torch.ops.aten.div.Tensor(a_musa, div_musa)
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_div_tensor_inplace_sparse_coo(dtype):
    if dtype in [torch.int32, torch.int64]:
        pytest.skip("div_tensor_inplace not support int")
    shape, nnz = (7, 9), 14
    a_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    a_musa = coo_to_musa(a_cpu)
    div_cpu = _scalar_tensor(2.5, dtype=dtype, device="cpu")
    div_musa = _scalar_tensor(2.5, dtype=dtype, device="musa")

    torch.ops.aten.div_.Tensor(a_cpu, div_cpu)
    torch.ops.aten.div_.Tensor(a_musa, div_musa)
    assert_dense_close(a_cpu, a_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_div_out_sparse_coo(dtype):
    shape, nnz = (7, 9), 14
    a_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    a_musa = coo_to_musa(a_cpu)
    div_cpu = _scalar_tensor(2.5, dtype=dtype, device="cpu")
    div_musa = _scalar_tensor(2.5, dtype=dtype, device="musa")

    y_ref = torch.ops.aten.div.Tensor(a_cpu, div_cpu)
    out_cpu = torch.empty_like(y_ref)
    out_musa = torch.empty_like(coo_to_musa(y_ref))

    torch.ops.aten.div.out(a_cpu, div_cpu, out=out_cpu)
    torch.ops.aten.div.out(a_musa, div_musa, out=out_musa)
    assert_dense_close(y_ref, out_cpu)
    assert_dense_close(coo_to_musa(y_ref), out_musa)


@pytest.mark.skipif(MUSPARSE_LT_12000, reason="requires MUSPARSE_VERSION >= 12000")
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.parametrize("dtype", dtypes)
def test_div_tensor_mode_sparse_coo(rounding_mode, dtype):
    shape, nnz = (5, 8), 12
    a_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    a_musa = coo_to_musa(a_cpu)
    div_cpu = _scalar_tensor(4, dtype=dtype, device="cpu")
    div_musa = _scalar_tensor(4, dtype=dtype, device="musa")

    y_cpu = torch.ops.aten.div.Tensor_mode(a_cpu, div_cpu, rounding_mode=rounding_mode)
    y_musa = torch.ops.aten.div.Tensor_mode(
        a_musa, div_musa, rounding_mode=rounding_mode
    )
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.skipif(MUSPARSE_LT_12000, reason="requires MUSPARSE_VERSION >= 12000")
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.parametrize("dtype", dtypes)
def test_div_tensor_mode_inplace_sparse_coo(rounding_mode, dtype):
    shape, nnz = (5, 8), 12
    a_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    a_musa = coo_to_musa(a_cpu)
    div_cpu = _scalar_tensor(4, dtype=dtype, device="cpu")
    div_musa = _scalar_tensor(4, dtype=dtype, device="musa")

    expected_cpu = torch.ops.aten.div.Tensor_mode(
        a_cpu.clone(), div_cpu, rounding_mode=rounding_mode
    )
    expected_musa = torch.ops.aten.div.Tensor_mode(
        a_musa.clone(), div_musa, rounding_mode=rounding_mode
    )

    torch.ops.aten.div_.Tensor_mode(a_cpu, div_cpu, rounding_mode=rounding_mode)
    torch.ops.aten.div_.Tensor_mode(a_musa, div_musa, rounding_mode=rounding_mode)
    assert_dense_close(a_cpu, expected_cpu)
    assert_dense_close(a_musa, expected_musa)


@pytest.mark.skipif(MUSPARSE_LT_12000, reason="requires MUSPARSE_VERSION >= 12000")
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.parametrize("dtype", dtypes)
def test_div_out_mode_sparse_coo(rounding_mode, dtype):
    shape, nnz = (5, 8), 12
    a_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    a_musa = coo_to_musa(a_cpu)
    div_cpu = _scalar_tensor(4, dtype=dtype, device="cpu")
    div_musa = _scalar_tensor(4, dtype=dtype, device="musa")

    y_ref = torch.ops.aten.div.Tensor_mode(a_cpu, div_cpu, rounding_mode=rounding_mode)
    out_cpu = torch.empty_like(y_ref)
    out_musa = torch.empty_like(coo_to_musa(y_ref))

    torch.ops.aten.div.out_mode(
        a_cpu, div_cpu, rounding_mode=rounding_mode, out=out_cpu
    )
    torch.ops.aten.div.out_mode(
        a_musa, div_musa, rounding_mode=rounding_mode, out=out_musa
    )
    assert_dense_close(y_ref, out_cpu)
    assert_dense_close(coo_to_musa(y_ref), out_musa)
