"""
Test sparse CSR/CSC conversion ops dispatch.
Covers `torch.ops.aten._to_sparse_csr` / `_to_sparse_csc`with different input
layouts to exercise the dispatch entries in
`torch_musa/csrc/aten/ops/musa_functions.yaml`:
- PrivateUse1: dense -> sparse (CSR/CSC)
- SparsePrivateUse1: COO -> sparse (CSR/CSC)
- SparseCsrPrivateUse1: compressed (CSR) -> sparse (CSR/CSC)
"""

# pylint: disable=missing-function-docstring, unused-import
import numpy as np
import pytest
import torch
from utils import make_dense_tensor


dtypes = [torch.float32, torch.float16, torch.int32, torch.int64, torch.bfloat16]
m_arr = [512]
n_arr = [1024]


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
def test_dense_to_sparse_csr(m, n, dtype):
    device = "musa"
    dense = make_dense_tensor(m, n, dtype)

    csr = dense.to_sparse_csr()

    assert csr.layout == torch.sparse_csr
    assert csr.device.type == device

    dense_back = csr.to_dense()
    torch.testing.assert_close(dense_back.cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
def test_coo_to_sparse_csr(m, n, dtype):
    device = "musa"
    dense = make_dense_tensor(m, n, dtype)

    coo = dense.to_sparse()
    csr = coo.to_sparse_csr()

    assert csr.layout == torch.sparse_csr
    assert csr.device.type == device

    dense_back = csr.to_dense()
    torch.testing.assert_close(dense_back.cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_compressed_to_sparse_csr(m, n, dtype):
    dense = make_dense_tensor(m, n, dtype)

    csc = dense.to_sparse_csc()
    out_from_csc = csc.to_sparse_csr()
    assert out_from_csc.layout == torch.sparse_csr
    torch.testing.assert_close(out_from_csc.to_dense().cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
def test_dense_to_sparse_csc(m, n, dtype):
    device = "musa"
    dense = make_dense_tensor(m, n, dtype)

    csc = dense.to_sparse_csc()

    assert csc.layout == torch.sparse_csc
    assert csc.device.type == device

    dense_back = csc.to_dense()
    torch.testing.assert_close(dense_back.cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
def test_coo_to_sparse_csc(m, n, dtype):
    device = "musa"
    dense = make_dense_tensor(m, n, dtype)

    coo = dense.to_sparse()

    csc = coo.to_sparse_csc()

    assert csc.layout == torch.sparse_csc
    assert csc.device.type == device

    dense_back = csc.to_dense()
    torch.testing.assert_close(dense_back.cpu(), dense.cpu())


@pytest.mark.parametrize("m", m_arr)
@pytest.mark.parametrize("n", n_arr)
@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_compressed_to_sparse_csc(m, n, dtype):
    dense = make_dense_tensor(m, n, dtype)

    csr = dense.to_sparse_csr()
    out_from_csr = csr.to_sparse_csc()

    assert out_from_csr.layout == torch.sparse_csc

    dense_back = out_from_csr.to_dense()
    torch.testing.assert_close(dense_back.cpu(), dense.cpu())
