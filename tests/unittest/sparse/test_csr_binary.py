"""Sparse CSR Binary tests"""

# pylint: disable=W0105, E0011, C0116, R1735
import pytest
import torch

from utils import make_csr, csr_to_musa, assert_csr_close, assert_dense_close

dtypes = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "case",
    [
        dict(name="add", out=True, inplace="add_"),
        dict(name="mul", out=True, inplace="mul_"),
    ],
    ids=lambda c: c["name"],
)
def test_csr_binary_tensor(dtype, case):
    name = case["name"]
    out = case.get("out", False)
    inplace = case.get("inplace", None)
    if name == "add":
        pytest.skip("musparse not support yet")

    shape = (2, 3)
    nnz = 3
    crow = torch.tensor([0, 2, 3], dtype=torch.int64)
    col = torch.tensor([0, 2, 1], dtype=torch.int64)
    val = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    a_musa = torch.sparse_csr_tensor(crow, col, val, size=shape, device="musa")
    b_musa = torch.randn(shape, dtype=dtype, device="musa")

    op = getattr(torch.ops.aten, name)

    # functional
    y_musa = op(a_musa, b_musa)
    if case["name"] == "mul":
        y_ref = a_musa.to_dense() * b_musa
    else:
        y_ref = a_musa.to_dense() + b_musa
    assert_dense_close(y_ref, y_musa.to_dense())

    # out
    if out:
        assert hasattr(op, "out")
        y_musa = make_csr(shape, nnz, dtype=dtype, device="musa")
        op.out(a_musa, b_musa, out=y_musa)
        assert_dense_close(y_ref, y_musa.to_dense())

    # inplace
    if inplace:
        inplace_op = getattr(torch.ops.aten, inplace)
        inplace_op(a_musa, b_musa)
        assert_dense_close(y_ref, y_musa.to_dense())


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "case",
    [
        dict(name="mul", scalar=-2.0, out=False, inplace="mul_"),
    ],
    ids=lambda c: c["name"],
)
def test_csr_binary_scalar(dtype, case):
    name = case["name"]
    inplace = case.get("inplace", None)
    scalar = case["scalar"]
    out = case.get("out", False)

    shape = (6, 10)
    nnz = 15
    a_cpu = make_csr(shape, nnz, dtype=dtype, device="cpu")
    a_musa = csr_to_musa(a_cpu)

    # functional
    op = getattr(torch.ops.aten, name)
    y_cpu = op.Scalar(a_cpu, scalar)
    y_musa = op.Scalar(a_musa, scalar)
    assert_csr_close(y_cpu, y_musa)

    # out
    if out:
        y_musa = make_csr(shape, nnz, dtype=dtype, device="musa")
        op.out(a_musa, scalar, out=y_musa)
        assert_csr_close(y_cpu, y_musa)

    # inplace
    if inplace:
        inplace_op = getattr(torch.ops.aten, inplace)
        # inplace_op = torch.ops.aten.mul_.Scalar
        inplace_op(a_cpu, scalar)
        inplace_op(a_musa, scalar)
        assert_csr_close(a_cpu, a_musa)
