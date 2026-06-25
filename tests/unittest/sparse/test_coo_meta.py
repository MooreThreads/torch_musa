"""Sparse COO meta op tests."""

# pylint: disable=missing-function-docstring, redefined-outer-name, C0103
import pytest
import torch

from utils import (
    make_coo,
    coo_to_musa,
)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (6, 10),  # sparse_dim=2, dense_dim=0
        (6, 10, 3),  # sparse_dim=2, dense_dim=1
        (6, 10, 2, 4),  # sparse_dim=2, dense_dim=2
    ],
)
def test_coo_dimI_dimV(dtype, shape):
    nnz = 12
    x_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)

    # _dimI: sparse_dim; _dimV: dense_dim
    dimI_cpu = torch.ops.aten._dimI(x_cpu)
    dimI_musa = torch.ops.aten._dimI(x_musa)
    assert int(dimI_cpu) == int(dimI_musa)
    assert int(dimI_cpu) == x_cpu.sparse_dim()

    dimV_cpu = torch.ops.aten._dimV(x_cpu)
    dimV_musa = torch.ops.aten._dimV(x_musa)
    assert int(dimV_cpu) == int(dimV_musa)
    assert int(dimV_cpu) == x_cpu.dense_dim()
