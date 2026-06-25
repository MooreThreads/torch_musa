"""
Unittest for sparse-related dispatch added in musa_functions.yaml (git diff).

Covers:
- ccol_indices / row_indices: SparseCsrPrivateUse1 on CSC
"""

# pylint: disable=missing-function-docstring
import numpy as np
import pytest
import torch

DEVICE = "musa"


def _make_dense(m, n, dtype, device=DEVICE):
    shape = (m, n)
    indices_x = []
    indices_y = []
    values = []
    for i in range(shape[0]):
        min_size = min(shape[0], max(1, shape[0] // 2))
        indices_x.extend([i] * min_size)
        indices_y.extend(np.random.randint(0, shape[1], size=min_size).tolist())
        values.extend([1] * min_size)
    indices = torch.tensor([indices_x, indices_y], dtype=torch.int64, device=device)
    values_t = torch.tensor(values, dtype=dtype, device=device)
    coo = torch.sparse_coo_tensor(indices, values_t, shape, device=device)
    return coo.to_dense()


@pytest.mark.parametrize("dtype", [torch.float32])
def test_ccol_indices_sparse_csc_dispatch(dtype):
    dense = _make_dense(8, 8, dtype)
    csc = dense.to_sparse_csc()
    ccol = csc.ccol_indices()
    assert ccol.device.type == DEVICE
    assert ccol.dim() == 1
    assert ccol.numel() == csc.size(1) + 1


@pytest.mark.parametrize("dtype", [torch.float32])
def test_row_indices_sparse_csc_dispatch(dtype):
    dense = _make_dense(8, 8, dtype)
    csc = dense.to_sparse_csc()
    row_idx = csc.row_indices()
    assert row_idx.device.type == DEVICE
    assert row_idx.dim() == 1
    assert row_idx.numel() == csc._nnz()
