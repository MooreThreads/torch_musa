"""
Test sparse BSR/BSC conversion ops dispatch.
Covers `torch.ops.aten._to_sparse_bsr` / `_to_sparse_bsc`with different input
layouts to exercise the dispatch entries in
`torch_musa/csrc/aten/ops/musa_functions.yaml`:
- PrivateUse1: dense -> sparse (BSR/BSC)
- SparsePrivateUse1: COO -> sparse (BSR/BSC)
- SparseCsrPrivateUse1: compressed (CSR) -> sparse (BSR/BSC)
"""

import numpy as np
import pytest
import torch


dtypes = [torch.float32, torch.float16, torch.int32, torch.int64, torch.bfloat16]
m_arr = [1024]
n_arr = [512]
block_sizes = [(1, 1), (4, 1), (1, 4)]


def _make_dense_tensor(m: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    rng = np.random.default_rng(seed=1234 + m * 1000 + n)
    min_sizez_per_row = max(1, m // 2)
    x_arr, y_arr, v_arr = [], [], []
    for i in range(m):
        min_size = int(rng.integers(low=0, high=min_sizez_per_row + 1))
        if min_size == 0:
            continue
        cols = rng.integers(low=0, high=n, size=min_size, dtype=np.int64).tolist()
        x_arr.extend([i] * min_size)
        y_arr.extend(cols)
        if dtype.is_floating_point:
            vals = (
                rng.normal(loc=0.0, scale=1.0, size=min_size)
                .astype(np.float32)
                .tolist()
            )
        else:
            vals = rng.integers(low=-8, high=8, size=min_size, dtype=np.int64).tolist()
        v_arr.extend(vals)
    if len(v_arr) == 0:
        x_arr, y_arr, v_arr = [0], [0], [1]
    indices = torch.tensor([x_arr, y_arr], dtype=torch.int64, device="musa")
    values = torch.tensor(v_arr, dtype=dtype, device="musa")
    coo = torch.sparse_coo_tensor(indices, values, (m, n), device="musa")
    return coo.to_dense()


def _skip_if_block_incompatible(m: int, n: int, blocksize):
    block_r, block_c = blocksize
    if m % block_r != 0 or n % block_c != 0:
        pytest.skip(f"incompatible blocksize={blocksize} for shape={(m, n)}")


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("blocksize", block_sizes)
def test_dense_to_sparse_bsr(m, n, dtype, blocksize):
    """test for dense_to_sparse_bsr"""
    _skip_if_block_incompatible(m, n, blocksize)
    dense = _make_dense_tensor(m, n, dtype)
    out = torch.ops.aten._to_sparse_bsr(dense, blocksize)
    assert out.layout == torch.sparse_bsr
    assert out.device.type == "musa"
    torch.testing.assert_close(out.to_dense().cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("blocksize", block_sizes)
def test_coo_to_sparse_bsr(m, n, dtype, blocksize):
    """test for coo_to_sparse_bsr"""
    _skip_if_block_incompatible(m, n, blocksize)
    dense = _make_dense_tensor(m, n, dtype)
    coo = dense.to_sparse()
    out = torch.ops.aten._to_sparse_bsr(coo, blocksize)
    assert out.layout == torch.sparse_bsr
    assert out.device.type == "musa"
    torch.testing.assert_close(out.to_dense().cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("blocksize", block_sizes)
def test_sparse_compressed_to_sparse_bsr(m, n, dtype, blocksize):
    """test for sparse_compressed_to_sparse_bsr"""
    _skip_if_block_incompatible(m, n, blocksize)
    dense = _make_dense_tensor(m, n, dtype)
    csr = dense.to_sparse_csr()
    out = torch.ops.aten._to_sparse_bsr(csr, blocksize)
    assert out.layout == torch.sparse_bsr
    assert out.device.type == "musa"
    torch.testing.assert_close(out.to_dense().cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("blocksize", block_sizes)
def test_dense_to_sparse_bsc(m, n, dtype, blocksize):
    """test for dense_to_sparse_bsc"""
    _skip_if_block_incompatible(m, n, blocksize)
    dense = _make_dense_tensor(m, n, dtype)
    out = torch.ops.aten._to_sparse_bsc(dense, blocksize)
    assert out.layout == torch.sparse_bsc
    assert out.device.type == "musa"
    torch.testing.assert_close(out.to_dense().cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("blocksize", block_sizes)
def test_coo_to_sparse_bsc(m, n, dtype, blocksize):
    """test for coo_to_sparse_bsc"""
    _skip_if_block_incompatible(m, n, blocksize)
    dense = _make_dense_tensor(m, n, dtype)
    coo = dense.to_sparse()
    out = torch.ops.aten._to_sparse_bsc(coo, blocksize)
    assert out.layout == torch.sparse_bsc
    assert out.device.type == "musa"
    torch.testing.assert_close(out.to_dense().cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("blocksize", block_sizes)
def test_sparse_compressed_to_sparse_bsc(m, n, dtype, blocksize):
    """test for sparse_compressed_to_sparse_bsc"""
    _skip_if_block_incompatible(m, n, blocksize)
    dense = _make_dense_tensor(m, n, dtype)
    csr = dense.to_sparse_csr()
    out = torch.ops.aten._to_sparse_bsc(csr, blocksize)
    assert out.layout == torch.sparse_bsc
    assert out.device.type == "musa"
    torch.testing.assert_close(out.to_dense().cpu(), dense.cpu())
