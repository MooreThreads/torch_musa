"""Unary sparsecsr op tests."""

# pylint: disable=R1735, C0103, C0411, R1735, C0116
import pytest
import torch
from torch_musa import testing

from utils import (
    make_csr,
    csr_to_musa,
    make_coo,
    coo_to_musa,
    assert_csr_close,
)

dtypes = [torch.float32, torch.float16, torch.bfloat16]


def prepare_nan_to_num(v):
    assert v.numel() >= 3
    v[0] = float("nan")
    v[1] = float("inf")
    v[2] = float("-inf")
    return v


CASES = [
    dict(name="abs", out=True, inplace="abs_"),
    dict(
        name="asin",
        out=True,
        inplace="asin_",
        sanitize=lambda v: v.uniform_(-1.0, 1.0),
    ),
    dict(name="asinh", out=True, inplace="asinh_"),
    dict(name="atan", out=True, inplace="atan_"),
    dict(
        name="atanh",
        out=True,
        inplace="atanh_",
        sanitize=lambda v: v.uniform_(-0.9, 0.9),
    ),
    dict(name="ceil", out=True, inplace="ceil_"),
    dict(name="deg2rad", out=True, inplace="deg2rad_"),
    dict(name="erf", out=True, inplace="erf_"),
    dict(
        name="erfinv",
        out=True,
        inplace="erfinv_",
        sanitize=lambda v: v.uniform_(-0.9, 0.9),
    ),
    dict(name="expm1", out=True, inplace="expm1_"),
    dict(name="floor", out=True, inplace="floor_"),
    dict(name="frac", out=True, inplace="frac_"),
    dict(
        name="log1p", out=True, inplace="log1p_", sanitize=lambda v: v.clamp(min=-0.9)
    ),
    dict(name="round", out=True, inplace="round_"),
    dict(name="rad2deg", out=True, inplace="rad2deg_"),
    dict(name="sign", out=True, inplace="sign_"),
    dict(name="sgn", out=True, inplace="sgn_"),
    dict(name="sin", out=True, inplace="sin_"),
    dict(name="sinh", out=True, inplace="sinh_"),
    dict(
        name="sqrt", out=True, inplace="sqrt_", sanitize=lambda v: v.uniform_(0.0, 4.0)
    ),
    dict(
        name="tan", out=True, inplace="tan_", sanitize=lambda v: v.uniform_(-1.0, 1.0)
    ),
    dict(name="tanh", out=True, inplace="tanh_"),
    dict(name="trunc", out=True, inplace="trunc_"),
    dict(name="relu", out=False, inplace="relu_"),
    dict(name="neg", out=True, inplace="neg_"),
    # COALESCED_UNARY_UFUNC_NO_INPLACE
    dict(name="signbit", out=True, inplace=None),
    dict(name="isneginf", out=True, inplace=None),
    dict(name="isposinf", out=True, inplace=None),
    # COALESCED_UNARY_UFUNC_FUNCTIONAL
    dict(name="isnan", out=False, inplace=None),
    dict(name="isinf", out=False, inplace=None),
]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("case", CASES, ids=lambda s: s["name"])
def test_csr_unary_operator_overloads(dtype, case):
    name = case["name"]
    out = case.get("out", False)
    inplace = case.get("inplace", None)
    sanitize = case.get("sanitize", None)

    x_cpu = make_csr((8, 12), 20, dtype=dtype, device="cpu")
    if sanitize is not None:
        v = sanitize(x_cpu.values().clone())
        x_cpu = torch.sparse_csr_tensor(
            x_cpu.crow_indices(), x_cpu.col_indices(), v, x_cpu.shape, dtype=dtype
        )
    x_musa = csr_to_musa(x_cpu)

    # 1) functional
    op = getattr(torch.ops.aten, name)
    y_cpu = op(x_cpu)
    y_musa = op(x_musa)
    assert_csr_close(y_cpu, y_musa)

    # 2) out overload (if requested)
    if out:
        assert hasattr(op, "out")
        y_musa.values().fill_(0)
        op.out(x_musa, out=y_musa)
        assert_csr_close(y_cpu, y_musa)

    # 3) inplace overload (if requested)
    if inplace is not None:
        inplace_op = getattr(torch.ops.aten, inplace)
        inplace_op(x_cpu)
        inplace_op(x_musa)
        assert_csr_close(x_cpu, x_musa)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("threshold", [-0.5, 0.0, 0.5])
def test_csr_threshold_backward(dtype, threshold):
    shape = (6, 7)
    nnz = 10

    self_cpu = make_csr(shape, nnz, dtype=dtype, device="cpu")
    grad_values = torch.randn_like(self_cpu.values())
    grad_cpu = torch.sparse_csr_tensor(
        self_cpu.crow_indices(),
        self_cpu.col_indices(),
        grad_values,
        self_cpu.shape,
        dtype=dtype,
        device="cpu",
    )

    self_musa = csr_to_musa(self_cpu)
    grad_musa = csr_to_musa(grad_cpu)

    # functional
    op = torch.ops.aten.threshold_backward
    y_cpu = op(grad_cpu, self_cpu, threshold)
    y_musa = op(grad_musa, self_musa, threshold)
    assert_csr_close(y_cpu, y_musa)

    # out
    op_out = torch.ops.aten.threshold_backward.grad_input
    y_cpu.values().zero_()
    y_musa.values().zero_()

    op_out(grad_cpu, self_cpu, threshold, grad_input=y_cpu)
    op_out(grad_musa, self_musa, threshold, grad_input=y_musa)
    assert_csr_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_csr_clone_zero_copy(dtype):
    x_cpu = make_csr((4, 8), 10, dtype=dtype, device="cpu")
    x_musa = csr_to_musa(x_cpu)
    clone_musa = x_musa.clone()
    cmp = testing.DefaultComparator()
    assert cmp(clone_musa.values().cpu(), x_musa.values().cpu())

    z = x_musa.clone()
    z.zero_()
    assert z._nnz() == 0

    y_musa = torch.sparse_csr_tensor(
        x_musa.crow_indices(),
        x_musa.col_indices(),
        torch.empty_like(x_musa.values()),
        size=x_musa.size(),
        device=x_musa.device,
    )
    y_musa.copy_(x_musa)
    assert cmp(x_musa.values().cpu(), y_musa.values().cpu())


@pytest.mark.parametrize("dtype", dtypes)
def test_csr_mask(dtype):
    a_cpu = make_coo((6, 8), 10, dtype=dtype, device="cpu")
    mask_cpu = make_csr((6, 8), 6, dtype=dtype, device="cpu")
    a_musa = coo_to_musa(a_cpu)
    mask_musa = csr_to_musa(mask_cpu)

    y_cpu = a_cpu.sparse_mask(mask_cpu)
    y_musa = a_musa.sparse_mask(mask_musa)
    assert_csr_close(y_cpu, y_musa)
