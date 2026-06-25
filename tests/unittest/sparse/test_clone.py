"""
Unittest for sparse-related dispatch added in musa_functions.yaml (git diff).

Covers:
- clone: SparsePrivateUse1 (clone_sparse), SparseCsrPrivateUse1 (clone_sparse_compressed)
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


# ----- clone: SparsePrivateUse1, SparseCsrPrivateUse1 -----
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_clone_sparse_coo_dispatch(dtype):
    dense = _make_dense(16, 16, dtype)
    coo = dense.to_sparse().coalesce()
    out = coo.clone()
    assert out.layout == torch.sparse_coo
    assert out.device.type == DEVICE
    torch.testing.assert_close(out.to_dense().cpu(), dense.cpu())


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_clone_sparse_compressed_dispatch(dtype):
    dense = _make_dense(16, 16, dtype)
    csr = dense.to_sparse_csr()
    out = csr.clone()
    assert out.layout == torch.sparse_csr
    assert out.device.type == DEVICE
    torch.testing.assert_close(out.to_dense().cpu(), dense.cpu())
