"""Sparse COO dispatch coverage for newly added ops."""

# pylint: disable=missing-function-docstring, C0411, W0611
import pytest
import torch
import torch_musa
from torch_musa import testing

from utils import (
    make_coo,
    coo_to_musa,
    assert_dense_close,
    _try_call,
    _require_op,
    MUSPARSE_LT_12000,
)

dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]


def _shape_tensor(device="cpu"):
    return torch.tensor([4, 7], dtype=torch.int64, device=device)


@pytest.mark.parametrize("dtype", dtypes)
def test_unsqueeze_sparse_coo(dtype):
    x_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)
    y_cpu = torch.unsqueeze(x_cpu, 0)
    y_musa = torch.unsqueeze(x_musa, 0)
    assert y_musa.is_sparse
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_index_select_sparse_coo(dtype):
    x_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)
    idx = torch.tensor([0, 2], dtype=torch.long, device="cpu")
    idx_m = idx.to("musa")
    y_cpu = torch.index_select(x_cpu, 0, idx)
    y_musa = torch.index_select(x_musa, 0, idx_m)
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_coo_tensor_with_dims(dtype):
    op = _require_op("_sparse_coo_tensor_with_dims")
    y_cpu = _try_call(
        [
            lambda: op(2, 0, [4, 7], dtype=dtype, device="cpu"),
            lambda: op(
                2, 0, [4, 7], dtype=dtype, layout=torch.sparse_coo, device="cpu"
            ),
        ]
    )
    y_musa = _try_call(
        [
            lambda: op(2, 0, [4, 7], dtype=dtype, device="musa"),
            lambda: op(
                2, 0, [4, 7], dtype=dtype, layout=torch.sparse_coo, device="musa"
            ),
        ]
    )
    assert y_cpu.layout == torch.sparse_coo
    assert y_musa.layout == torch.sparse_coo
    assert y_musa.device.type == "musa"
    assert list(y_cpu.shape) == list(y_musa.shape)
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_resize_sparse_coo(dtype):
    op = _require_op("sparse_resize_")
    x_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)
    _try_call(
        [
            lambda: op(x_cpu, [4, 7], 2, 0),
            lambda: op(x_cpu, _shape_tensor("cpu"), 2, 0),
        ]
    )
    _try_call(
        [
            lambda: op(x_musa, [4, 7], 2, 0),
            lambda: op(x_musa, _shape_tensor("musa"), 2, 0),
        ]
    )
    assert x_musa.is_sparse
    assert list(x_cpu.shape) == list(x_musa.shape)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_broadcast_to_sparse_coo(dtype):
    op = _require_op("_sparse_broadcast_to")
    x_cpu = make_coo((1, 7), 6, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)
    y_cpu = _try_call(
        [lambda: op(x_cpu, [4, 7]), lambda: op(x_cpu, _shape_tensor("cpu"))]
    )
    y_musa = _try_call(
        [lambda: op(x_musa, [4, 7]), lambda: op(x_musa, _shape_tensor("musa"))]
    )
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_permute_sparse_coo(dtype):
    x_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)
    y_cpu = x_cpu.permute(1, 0)
    y_musa = x_musa.permute(1, 0)
    assert y_musa.is_sparse
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_narrow_copy_sparse_coo(dtype):
    x_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)
    y_cpu = torch.narrow_copy(x_cpu, 0, 1, 2)
    y_musa = torch.narrow_copy(x_musa, 0, 1, 2)
    assert y_musa.is_sparse
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.skipif(MUSPARSE_LT_12000, reason="requires MUSPARSE_VERSION >= 12000")
@pytest.mark.skipif(testing.get_musa_arch() < 31, reason="requires mp >= 31")
def test_sparse_sparse_matmul_sparse_coo(dtype):
    if dtype not in [torch.float32]:
        pytest.skip("sparse_matmul only support torch.float32")
    op = _require_op("_sparse_sparse_matmul")
    a_cpu = make_coo((4, 6), 8, dtype=dtype, device="cpu")
    b_cpu = make_coo((6, 5), 8, dtype=dtype, device="cpu")
    a_musa = coo_to_musa(a_cpu)
    b_musa = coo_to_musa(b_cpu)
    y_cpu = op(a_cpu, b_cpu)
    y_musa = op(a_musa, b_musa)
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_sum_backward_sparse_coo(dtype):
    if dtype not in [torch.float16, torch.float32]:
        pytest.skip(
            "_sparse_sum_backward_cuda only support [torch.float16, torch.float32]"
        )
    op = _require_op("_sparse_sum_backward")
    x_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    grad_cpu = make_coo((4, 7), 4, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)
    grad_musa = coo_to_musa(grad_cpu)
    y_cpu = op(grad_cpu, x_cpu, [0])
    y_musa = op(grad_musa, x_musa, [0])
    assert y_musa.is_sparse
    assert_dense_close(y_cpu, y_musa)
    y_cpu = op(grad_cpu, x_cpu, [1])
    y_musa = op(grad_musa, x_musa, [1])
    assert y_musa.is_sparse
    assert_dense_close(y_cpu, y_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_softmax_and_backward_sparse_coo(dtype):
    if dtype not in [torch.float32]:
        pytest.skip("softmax only support torch.float32")
    x_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)
    y_cpu = torch.sparse.softmax(x_cpu, dim=1)
    y_musa = torch.sparse.softmax(x_musa, dim=1)
    assert_dense_close(y_cpu, y_musa)

    op_bw = _require_op("_sparse_softmax_backward_data")
    grad_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    grad_musa = coo_to_musa(grad_cpu)
    bw_cpu = _try_call(
        [
            lambda: op_bw(grad_cpu, y_cpu, 1, x_cpu),
            lambda: op_bw(grad_cpu, y_cpu, 1, x_cpu, None),
        ]
    )
    bw_musa = _try_call(
        [
            lambda: op_bw(grad_musa, y_musa, 1, x_musa),
            lambda: op_bw(grad_musa, y_musa, 1, x_musa, None),
        ]
    )
    assert_dense_close(bw_cpu, bw_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_log_softmax_and_backward_sparse_coo(dtype):
    if dtype not in [torch.float32]:
        pytest.skip("log_sofmax only support torch.float32")
    x_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    x_musa = coo_to_musa(x_cpu)
    y_cpu = torch.sparse.log_softmax(x_cpu, dim=1)
    y_musa = torch.sparse.log_softmax(x_musa, dim=1)
    assert_dense_close(y_cpu, y_musa)

    op_bw = _require_op("_sparse_log_softmax_backward_data")
    grad_cpu = make_coo((4, 7), 8, dtype=dtype, device="cpu")
    grad_musa = coo_to_musa(grad_cpu)
    bw_cpu = _try_call(
        [
            lambda: op_bw(grad_cpu, y_cpu, 1, x_cpu),
            lambda: op_bw(grad_cpu, y_cpu, 1, x_cpu, None),
        ]
    )
    bw_musa = _try_call(
        [
            lambda: op_bw(grad_musa, y_musa, 1, x_musa),
            lambda: op_bw(grad_musa, y_musa, 1, x_musa, None),
        ]
    )
    assert_dense_close(bw_cpu, bw_musa)


@pytest.mark.parametrize("dtype", dtypes)
def test_sspaddmm_out_sparse_coo(dtype):
    pytest.skip("NYI: CUDA sspaddmm is not implemented")
    if dtype not in [torch.float32]:
        pytest.skip("sspaddmm only support torch.float32")
    a_cpu = make_coo((4, 5), 6, dtype=dtype, device="cpu")
    b_cpu = make_coo((4, 3), 6, dtype=dtype, device="cpu")
    c_cpu = torch.randn((3, 5), dtype=dtype)
    a_musa = coo_to_musa(a_cpu)
    b_musa = coo_to_musa(b_cpu)
    c_musa = c_cpu.to("musa")

    out_cpu = torch.sparse_coo_tensor(
        a_cpu.indices(), a_cpu.values().clone(), a_cpu.shape, dtype=dtype, device="cpu"
    ).coalesce()
    out_musa = coo_to_musa(out_cpu)
    torch.sspaddmm(a_cpu, b_cpu, c_cpu, out=out_cpu)
    torch.sspaddmm(a_musa, b_musa, c_musa, out=out_musa)
    assert out_musa.is_sparse
    assert_dense_close(out_cpu, out_musa)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.skipif(MUSPARSE_LT_12000, reason="requires MUSPARSE_VERSION >= 12000")
def test_hspmm_and_out_sparse_coo(dtype):
    if dtype not in [torch.float32]:
        pytest.skip("addmm_sparse_cuda only support torch.float32")
    sparse_cpu = make_coo((4, 6), 8, dtype=dtype, device="cpu")
    dense_cpu = torch.randn((6, 5), dtype=dtype)
    sparse_musa = coo_to_musa(sparse_cpu)
    dense_musa = dense_cpu.to("musa")

    y_cpu = torch.hspmm(sparse_cpu, dense_cpu)
    y_musa = torch.hspmm(sparse_musa, dense_musa)
    assert y_musa.is_sparse
    assert_dense_close(y_cpu, y_musa)

    out_cpu = y_cpu.clone()
    out_musa = coo_to_musa(out_cpu.coalesce())
    op = _require_op("hspmm")
    _try_call(
        [
            lambda: op.out(sparse_cpu, dense_cpu, out=out_cpu),
            lambda: torch.hspmm(sparse_cpu, dense_cpu, out=out_cpu),
        ]
    )
    _try_call(
        [
            lambda: op.out(sparse_musa, dense_musa, out=out_musa),
            lambda: torch.hspmm(sparse_musa, dense_musa, out=out_musa),
        ]
    )
    assert_dense_close(out_cpu, out_musa)
