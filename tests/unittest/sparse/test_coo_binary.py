"""Sparse COO Binary tests"""

# pylint: disable=W0105, E0011, C0116, R1735
import pytest
import torch

from utils import (
    make_coo,
    coo_to_musa,
    assert_dense_close,
)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "case",
    [
        dict(name="add", out=True, inplace="add_"),
        dict(name="sub", out=True, inplace="sub_"),
        dict(name="mul", out=True, inplace="mul_"),
    ],
    ids=lambda c: c["name"],
)
def test_coo_binary_tensor(dtype, case):
    name = case["name"]
    out = case.get("out", False)
    inplace = case.get("inplace", None)

    shape = (6, 10)
    nnz = 15
    a_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    b_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    a_musa = coo_to_musa(a_cpu)
    b_musa = coo_to_musa(b_cpu)
    op = getattr(torch.ops.aten, name)

    # functional
    y_cpu = op(a_cpu, b_cpu)
    y_musa = op(a_musa, b_musa)
    assert_dense_close(y_cpu, y_musa)

    # out
    if out:
        assert hasattr(op, "out")
        y_musa = make_coo(shape, nnz, dtype=dtype, device="musa")
        op.out(a_musa, b_musa, out=y_musa)
        assert_dense_close(y_cpu, y_musa)

    # inplace
    if inplace:
        inplace_op = getattr(torch.ops.aten, inplace)
        inplace_op(a_cpu, b_cpu)
        inplace_op(a_musa, b_musa)
        assert_dense_close(a_cpu, a_musa)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "case",
    [
        dict(name="mul", scalar=-2.0, out=True, inplace="mul_"),
        dict(name="div", scalar=2.0, out=True, inplace="div_"),
    ],
    ids=lambda c: c["name"],
)
def test_coo_binary_scalar(dtype, case):
    name = case["name"]
    inplace = case.get("inplace", None)
    scalar = case["scalar"]
    out = case.get("out", False)

    shape = (6, 10)
    nnz = 15
    a_cpu = make_coo(shape, nnz, dtype=dtype, device="cpu")
    a_cpu_dense = a_cpu.to_dense()
    a_musa = coo_to_musa(a_cpu)

    # functional
    op = getattr(torch.ops.aten, name)
    y_cpu_dense = op(a_cpu_dense, scalar)
    y_musa = op(a_musa, scalar)
    assert_dense_close(y_cpu_dense, y_musa)

    # out
    if out:
        y_musa = make_coo(shape, nnz, dtype=dtype, device="musa")
        op.out(a_musa, scalar, out=y_musa)
        assert_dense_close(y_cpu_dense, y_musa)

    # inplace
    if inplace:
        inplace_op = getattr(torch.ops.aten, name + "_")
        inplace_op(a_cpu_dense, scalar)
        inplace_op(a_musa, scalar)
        assert_dense_close(a_cpu_dense, a_musa)
