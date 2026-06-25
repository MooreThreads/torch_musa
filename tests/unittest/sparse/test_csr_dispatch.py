"""Sparse CSR dispatch coverage for listed ops (CPU vs MUSA)."""

# pylint: disable=missing-function-docstring, C0411, W0611
import pytest
import torch
import torch_musa
from torch_musa import testing

from utils import (
    make_csr,
    csr_to_musa,
    assert_dense_close,
    assert_csr_close,
    _require_op,
    _try_call,
    MUSPARSE_LT_12000,
)

dtypes = [torch.float32, torch.float16, torch.bfloat16]


def _make_csr_inputs(dtype=torch.float32):
    x_cpu = make_csr((6, 8), 12, dtype=dtype, device="cpu")
    x_musa = csr_to_musa(x_cpu)
    return x_cpu, x_musa


@pytest.mark.parametrize("dtype", dtypes)
def test_normal_sparse_csr_(dtype):
    nnz = 10000
    crow = torch.tensor([0, nnz])
    col = torch.arange(nnz)
    x_cpu = torch.sparse_csr_tensor(crow, col, torch.zeros(nnz, dtype=dtype))

    x_musa = torch.sparse_csr_tensor(
        crow.musa(), col.musa(), torch.zeros(nnz, dtype=dtype, device="musa")
    )

    x_cpu = torch.ops.aten.normal_(x_cpu, 0, 1)
    x_musa = torch.ops.aten.normal_(x_musa, 0, 1)

    cpu_vals = x_cpu.values()
    musa_vals = x_musa.values().cpu()

    # 只比统计特征
    torch.testing.assert_close(cpu_vals.mean(), musa_vals.mean(), atol=0.05, rtol=0.05)
    torch.testing.assert_close(cpu_vals.std(), musa_vals.std(), atol=0.05, rtol=0.05)


@pytest.mark.parametrize("dtype", dtypes)
def test_sum_sparse_csr(dtype):
    x_cpu, x_musa = _make_csr_inputs(dtype)
    y_cpu = torch.sum(x_cpu)
    y_musa = torch.sum(x_musa)
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_sum_out_sparse_csr(dtype):
    x_cpu, x_musa = _make_csr_inputs(dtype)
    op = _require_op("sum")
    out_cpu = torch.empty((), dtype=dtype, device="cpu")
    out_musa = torch.empty((), dtype=dtype, device="musa")
    _try_call(
        [
            lambda: op.out(x_cpu, out=out_cpu),
            lambda: op.out(x_cpu, dtype=dtype, out=out_cpu),
        ]
    )
    _try_call(
        [
            lambda: op.out(x_musa, out=out_musa),
            lambda: op.out(x_musa, dtype=dtype, out=out_musa),
        ]
    )
    assert_dense_close(out_cpu, out_musa)


def _make_batched_csr_baddbmm_inputs(dtype):
    """Return (self_csr, batch1, batch2) for baddbmm.out if batched CSR is supported."""
    n, k, m = 4, 5, 6
    batch1 = torch.randn(n, k, dtype=dtype)
    batch2 = torch.randn(m, k, dtype=dtype)
    self_dense = torch.randn(n, m, dtype=dtype)

    try:
        self_csr = self_dense.to_sparse_csr()
    except Exception as exc:
        raise RuntimeError(f"batched CSR not constructible: {exc}") from exc

    return self_csr, batch1, batch2


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.skipif(MUSPARSE_LT_12000, reason="requires MUSPARSE_VERSION >= 12000")
@pytest.mark.skipif(testing.get_musa_arch() < 31, reason="requires mp >= 31")
def test_baddbmm_out_sparse_csr(dtype):
    try:
        self_cpu, b1_cpu, b2_cpu = _make_batched_csr_baddbmm_inputs(dtype)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    csr_musa = csr_to_musa(self_cpu)
    b1_m = b1_cpu.to("musa")
    b2_m = b2_cpu.to("musa")

    out_m = torch.empty_like(b1_m)

    beta = 1.0
    alpha = 1.0
    torch.ops.aten.baddbmm.out(b1_m, csr_musa, b2_m, beta=beta, alpha=alpha, out=out_m)

    ref = beta * b1_m + alpha * torch.matmul(csr_musa.to_dense(), b2_m)

    assert_dense_close(out_m, ref)


@pytest.mark.parametrize("dtype", dtypes)
def test_empty_like_sparse_csr(dtype):
    x_cpu, x_musa = _make_csr_inputs(dtype)
    y_cpu = torch.empty_like(x_cpu)
    y_musa = torch.empty_like(x_musa)
    assert y_cpu.layout == torch.sparse_csr and y_musa.layout == torch.sparse_csr
    assert list(y_cpu.shape) == list(y_musa.shape)


@pytest.mark.parametrize("dtype", dtypes)
def test_angle_and_angle_out_sparse_csr(dtype):
    if dtype not in [torch.float32]:
        pytest.skip("angle_cuda only support torch.float32")
    x_cpu, x_musa = _make_csr_inputs(dtype)
    y_cpu = torch.angle(x_cpu)
    y_musa = torch.angle(x_musa)
    assert_csr_close(y_cpu, y_musa)

    op = _require_op("angle")
    out_cpu = torch.empty_like(y_cpu)
    out_musa = torch.empty_like(y_musa)
    _try_call([lambda: op.out(x_cpu, out=out_cpu)])
    _try_call([lambda: op.out(x_musa, out=out_musa)])
    assert_csr_close(out_cpu, out_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_conj_physical_sparse_csr(dtype):
    op = _require_op("_conj_physical")
    x_cpu, x_musa = _make_csr_inputs(dtype)
    y_cpu = op(x_cpu)
    y_musa = op(x_musa)
    assert_csr_close(y_cpu, y_musa)

    op = _require_op("conj_physical")
    op.out(x_cpu, out=y_cpu)
    op.out(x_musa, out=y_musa)
    assert_csr_close(y_cpu, y_musa)

    op = _require_op("conj_physical_")
    op(x_cpu)
    op(x_musa)
    assert_csr_close(x_cpu, x_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_select_copy_and_select_int_sparse_csr(dtype):
    torch.manual_seed(42)  # avoid duplicate col indices within a row
    x_cpu, x_musa = _make_csr_inputs(dtype)
    op_sc = _require_op("select_copy")
    op_sel = _require_op("select")

    y_cpu = _try_call(
        [
            lambda: op_sc.int(x_cpu, 0, 1),
            lambda: torch.select_copy(x_cpu, 0, 1),
        ]
    )
    y_musa = _try_call(
        [
            lambda: op_sc.int(x_musa, 0, 1),
            lambda: torch.select_copy(x_musa, 0, 1),
        ]
    )
    assert_dense_close(y_cpu, y_musa)

    z_cpu = _try_call(
        [lambda: op_sel.int(x_cpu, 0, 1), lambda: torch.select(x_cpu, 0, 1)]
    )
    z_musa = _try_call(
        [lambda: op_sel.int(x_musa, 0, 1), lambda: torch.select(x_musa, 0, 1)]
    )
    assert_dense_close(z_cpu, z_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_csr_sum_and_prod_dim_dtype(dtype):
    x_cpu, x_musa = _make_csr_inputs(dtype)

    s_cpu = torch.ops.aten._sparse_csr_sum.dim_dtype(x_cpu, [1], True, dtype=dtype)
    s_musa = torch.ops.aten._sparse_csr_sum.dim_dtype(x_musa, [1], True, dtype=dtype)
    assert_csr_close(s_cpu, s_musa)

    p_cpu = torch.ops.aten._sparse_csr_prod.dim_dtype(x_cpu, [1], True, dtype=dtype)
    p_musa = torch.ops.aten._sparse_csr_prod.dim_dtype(x_musa, [1], True, dtype=dtype)
    assert_csr_close(p_cpu, p_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_spsolve_sparse_csr(dtype):
    """_spsolve has no SparseCsrCPU backend"""
    pytest.skip("requires compiling PyTorch with CUDA cuDSS")
    op = _require_op("_spsolve")
    n = 5
    a_dense = torch.eye(n, dtype=dtype) * 2.0 + 0.1 * torch.randn(n, n, dtype=dtype)
    a_cpu = a_dense.to_sparse_csr()
    b_cpu = torch.randn(n, dtype=dtype)
    a_musa = csr_to_musa(a_cpu)
    b_musa = b_cpu.to("musa")
    a_fp32_musa = a_musa.to(torch.float32)
    b_fp32_musa = b_musa.to(torch.float32)

    y_musa = op(a_musa, b_musa)
    y_fp32_musa = op(a_fp32_musa, b_fp32_musa)
    assert_dense_close(y_musa, y_fp32_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_sampled_addmm_and_out_sparse_csr(dtype):
    if dtype not in [torch.float32]:
        pytest.skip("sparse_sampled_addmm only support torch.float32")
    op = _require_op("sparse_sampled_addmm")
    mat1_cpu = torch.randn((5, 7), dtype=dtype)
    mat2_cpu = torch.randn((7, 4), dtype=dtype)
    mask_dense = (torch.randn((5, 4), dtype=dtype) > 0).to(dtype)
    mask_cpu = mask_dense.to_sparse_csr()

    mat1_m = mat1_cpu.to("musa")
    mat2_m = mat2_cpu.to("musa")
    mask_m = csr_to_musa(mask_cpu)

    y_cpu = _try_call(
        [
            lambda: op(mask_cpu, mat1_cpu, mat2_cpu),
            lambda: op(mask_cpu, mat1_cpu, mat2_cpu, beta=0.0, alpha=1.0),
        ]
    )
    y_musa = _try_call(
        [
            lambda: op(mask_m, mat1_m, mat2_m),
            lambda: op(mask_m, mat1_m, mat2_m, beta=0.0, alpha=1.0),
        ]
    )
    assert_dense_close(y_cpu, y_musa)

    out_cpu = y_cpu.clone()
    out_m = csr_to_musa(out_cpu)
    op_out = _require_op("sparse_sampled_addmm")
    _try_call(
        [
            lambda: op_out.out(mask_cpu, mat1_cpu, mat2_cpu, out=out_cpu),
            lambda: op_out.out(
                mask_cpu, mat1_cpu, mat2_cpu, beta=0.0, alpha=1.0, out=out_cpu
            ),
        ]
    )
    _try_call(
        [
            lambda: op_out.out(mask_m, mat1_m, mat2_m, out=out_m),
            lambda: op_out.out(mask_m, mat1_m, mat2_m, beta=0.0, alpha=1.0, out=out_m),
        ]
    )
    assert_dense_close(out_cpu, out_m)
