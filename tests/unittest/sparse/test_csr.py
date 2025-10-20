"""Test sparse csr operators."""

# pylint: disable=missing-function-docstring, unused-import
import numpy as np
import pytest
import torch


def test_coo_to_csr():
    shape = (16, 512)
    indices_x = []
    indices_y = []
    values = []
    for i in range(shape[0]):
        n = np.random.randint(0, 16)
        indices_x.extend([i] * n)
        indices_y.extend(np.random.randint(0, shape[1], n))
        values.extend([np.random.randint(0, 16) for _ in range(n)])

    indices = torch.tensor([indices_x, indices_y])
    values = torch.tensor(values, dtype=torch.float32)
    indices_m = indices.musa()
    values_m = values.musa()

    coo = torch.sparse_coo_tensor(indices, values, shape)
    coo_musa = torch.sparse_coo_tensor(indices_m, values_m, shape)

    assert coo._nnz() == coo_musa._nnz()
    coo_coalesce, coo_musa_coalesce = coo.coalesce(), coo_musa.coalesce()
    assert torch.all(coo_coalesce.indices() == coo_musa_coalesce.indices().cpu())
    assert torch.all(coo_coalesce.values() == coo_musa_coalesce.values().cpu())

    csr = coo.to_sparse_csr()
    csr_musa = coo_musa.to_sparse_csr()
    assert csr._nnz() == csr_musa._nnz()
    assert torch.all(csr.crow_indices() == csr_musa.crow_indices().cpu())
    assert torch.all(csr.col_indices() == csr_musa.col_indices().cpu())
    assert torch.all(csr.values() == csr_musa.values().cpu())


def test_csr_to_coo():
    shape = (16, 512)
    cumsum_x = [0]
    indices_y = []
    values = []
    for _ in range(shape[0]):
        n = np.random.randint(0, 16)
        pre_num = cumsum_x[-1]
        cumsum_x.append(pre_num + n)
        indices_y.extend(np.random.randint(0, shape[1], n))
        values.extend([np.random.randint(0, 16) for _ in range(n)])

    crow_indices = torch.tensor(cumsum_x)
    col_indices = torch.tensor(indices_y)
    values = torch.tensor(values, dtype=torch.float32)

    crow_indices_m = crow_indices.musa()
    col_indices_m = col_indices.musa()
    values_m = values.musa()

    csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, shape)
    csr_musa = torch.sparse_csr_tensor(crow_indices_m, col_indices_m, values_m, shape)
    assert csr._nnz() == csr_musa._nnz()
    assert torch.all(csr.crow_indices() == csr_musa.crow_indices().cpu())
    assert torch.all(csr.col_indices() == csr_musa.col_indices().cpu())
    assert torch.all(csr.values() == csr_musa.values().cpu())

    coo = csr.to_sparse()
    coo_musa = csr_musa.to_sparse()
    coo_coalesce, coo_musa_coalesce = coo.coalesce(), coo_musa.coalesce()
    assert torch.all(coo_coalesce.indices() == coo_musa_coalesce.indices().cpu())
    assert torch.all(coo_coalesce.values() == coo_musa_coalesce.values().cpu())
