"""
Unittest for sparse-related dispatch added in musa_functions.yaml (git diff).

Covers:
- resize_: SparseCsrPrivateUse1 (resize_sparse_csr_)
- resize_as_sparse_: SparsePrivateUse1, SparseCsrPrivateUse1
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


# ----- resize_: SparseCsrPrivateUse1 -----
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_resize_sparse_csr_dispatch(dtype):
    dense = _make_dense(8, 16, dtype)
    csr = dense.to_sparse_csr()
    new_size = [4, 32]
    csr.resize_(new_size)
    assert csr.layout == torch.sparse_csr
    assert csr.device.type == DEVICE
    assert list(csr.shape) == new_size


# ----- resize_as_sparse_: SparsePrivateUse1, SparseCsrPrivateUse1 -----
@pytest.mark.parametrize("dtype", [torch.float32])
def test_resize_as_sparse_coo_dispatch(dtype):
    dense = _make_dense(4, 8, dtype)
    coo = dense.to_sparse()
    target = _make_dense(8, 10, dtype)
    target_coo = target.to_sparse()
    coo.resize_as_sparse_(target_coo)
    assert coo.layout == torch.sparse_coo
    assert coo.device.type == DEVICE
    assert list(coo.shape) == list(target_coo.shape)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_resize_as_sparse_compressed_dispatch(dtype):
    dense = _make_dense(8, 8, dtype)
    csr = dense.to_sparse_csr()
    target = _make_dense(4, 4, dtype)
    target_csr = target.to_sparse_csr()
    csr.resize_as_sparse_(target_csr)
    assert csr.layout == torch.sparse_csr
    assert csr.device.type == DEVICE
    assert list(csr.shape) == list(target_csr.shape)
