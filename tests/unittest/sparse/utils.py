"""Sparse COO test utilities."""

# pylint: disable=missing-function-docstring
import re
from functools import lru_cache
from pathlib import Path
import numpy as np
import pytest
import torch
from torch_musa import testing


@lru_cache
def get_musparse_version() -> int:
    # test_coo_dispatch.py -> sparse -> unittest -> tests -> repo_root
    repo_root = Path(__file__).resolve().parents[3]
    cmake_file = repo_root / "CMakeLists.txt"
    text = cmake_file.read_text(encoding="utf-8")
    m = re.search(r'set\s*\(\s*MUSPARSE_VERSION\s+"?(\d+)"?\s*\)', text)
    if not m:
        # 保守策略：读不到就按低版本处理，避免误跑不支持用例
        return 0
    return int(m.group(1))


MUSPARSE_LT_12000 = get_musparse_version() < 12000


def _require_op(name):
    if not hasattr(torch.ops.aten, name):
        pytest.skip("torch.ops.aten.{name} is unavailable in this torch build")
    return getattr(torch.ops.aten, name)


def _try_call(candidates):
    last_exc = None
    res = None
    for func in candidates:
        try:
            res = func()
        except (TypeError, RuntimeError, AttributeError) as exc:
            last_exc = exc
    if last_exc:
        pytest.skip("No compatible overload found for current torch build: {last_exc}")
    return res


def _rand_values_for_sparse(nnz, shape_tail, *, dtype, device):
    """Sparse value init: randn is invalid for integer dtypes (no normal kernel)."""
    tail = (nnz,) + tuple(shape_tail)
    if dtype.is_floating_point or getattr(dtype, "is_complex", False):
        return torch.randn(*tail, dtype=dtype, device=device)
    if dtype == torch.bool:
        return torch.randint(0, 2, tail, dtype=torch.bool, device=device)
    if dtype == torch.uint8:
        return torch.randint(1, 255, tail, dtype=torch.uint8, device=device)
    tmp = torch.randint(-100, 101, tail, dtype=torch.int64, device=device)
    return tmp.to(dtype=dtype)


def make_coo(shape, nnz, *, dtype=torch.float32, device="cpu"):
    assert len(shape) >= 2
    i = torch.randint(0, shape[0], (1, nnz), device=device)
    j = torch.randint(0, shape[1], (1, nnz), device=device)
    indices = torch.cat([i, j], dim=0)
    # values = torch.randn(nnz, *shape[2:], dtype=dtype, device=device)
    values = _rand_values_for_sparse(nnz, shape[2:], dtype=dtype, device=device)
    x = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)
    return x.coalesce()


def coo_to_musa(x_cpu):
    return torch.sparse_coo_tensor(
        x_cpu.indices().to("musa"),
        x_cpu.values().to("musa"),
        x_cpu.shape,
        dtype=x_cpu.dtype,
        device="musa",
    ).coalesce()


def assert_coo_close(cpu, musa):
    cpu = cpu.coalesce()
    musa = musa.coalesce()
    assert cpu.is_sparse and musa.is_sparse
    assert cpu.layout == musa.layout
    assert cpu.shape == musa.shape
    assert cpu._nnz() == musa._nnz()
    assert torch.equal(
        cpu.indices().to(torch.int64),
        musa.indices().cpu().to(torch.int64),
    )

    abs_diff = 1e-5
    rel_diff = 1e-5
    if cpu.dtype == torch.float32:
        abs_diff = 1e-5
        rel_diff = 1e-5
    elif cpu.dtype == torch.float16:
        abs_diff = 1e-3
        rel_diff = 1e-3
    elif cpu.dtype == torch.bfloat16:
        abs_diff = 8e-3
        rel_diff = 8e-3

    cmp = testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff)
    assert cmp(musa.values().cpu(), cpu.values())


def assert_dense_close(cpu, musa):
    if cpu.is_sparse or cpu.is_sparse_csr:
        dense_cpu = cpu.to_dense()
    else:
        dense_cpu = cpu
    if musa.is_sparse or musa.is_sparse_csr:
        dense_musa = musa.to_dense().cpu()
    else:
        dense_musa = musa.cpu()

    abs_diff = 1e-5
    rel_diff = 1e-5
    if cpu.dtype == torch.float32:
        abs_diff = 1e-5
        rel_diff = 1e-5
    elif cpu.dtype == torch.float16:
        abs_diff = 2e-3
        rel_diff = 2e-3
    elif cpu.dtype == torch.bfloat16:
        abs_diff = 8e-3
        rel_diff = 8e-3

    cmp = testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff)
    assert cmp(dense_musa, dense_cpu)


def make_dense_tensor(m, n, dtype):
    shape = (m, n)
    indices_x = []
    indices_y = []
    values = []
    for i in range(shape[0]):
        min_size = np.random.randint(0, m)
        indices_x.extend([i] * min_size)
        indices_y.extend(np.random.randint(0, shape[1], min_size))
        values.extend([np.random.randint(0, m) for _ in range(min_size)])

    indices = torch.tensor([indices_x, indices_y]).musa()
    values = torch.tensor(values, dtype=dtype).musa()

    coo = torch.sparse_coo_tensor(indices, values, shape)
    return coo.to_dense()


def make_csr(shape, nnz, *, dtype=torch.float32, device="cpu"):
    assert len(shape) >= 2
    rows, cols = shape[0], shape[1]

    row_indices = torch.randint(0, rows, (nnz,), device=device)
    row_indices, perm = torch.sort(row_indices)

    col_indices = torch.randint(0, cols, (nnz,), device=device)
    col_indices = col_indices[perm]

    # values = torch.randn(nnz, *shape[2:], dtype=dtype, device=device)
    values = _rand_values_for_sparse(nnz, shape[2:], dtype=dtype, device=device)
    values = values[perm]

    crow_indices = torch.zeros(rows + 1, dtype=torch.int64, device=device)
    counts = torch.bincount(row_indices, minlength=rows)
    crow_indices[1:] = torch.cumsum(counts, dim=0)

    csr = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, size=shape, dtype=dtype, device=device
    )
    return csr


def csr_to_musa(x_cpu):
    return torch.sparse_csr_tensor(
        x_cpu.crow_indices().to("musa"),
        x_cpu.col_indices().to("musa"),
        x_cpu.values().to("musa"),
        x_cpu.shape,
        dtype=x_cpu.dtype,
        device="musa",
    )


def assert_csr_close(cpu, musa):
    assert cpu.layout == torch.sparse_csr
    assert musa.layout == torch.sparse_csr

    assert cpu.shape == musa.shape
    assert cpu._nnz() == musa._nnz()

    # crow_indices
    assert torch.equal(
        cpu.crow_indices().to(torch.int64),
        musa.crow_indices().cpu().to(torch.int64),
    )

    assert torch.equal(
        cpu.col_indices().to(torch.int64),
        musa.col_indices().cpu().to(torch.int64),
    )
    abs_diff = 1e-5
    rel_diff = 1e-5
    if cpu.dtype == torch.float32:
        abs_diff = 1e-5
        rel_diff = 1e-5
    elif cpu.dtype == torch.float16:
        abs_diff = 1e-3
        rel_diff = 1e-3
    elif cpu.dtype == torch.bfloat16:
        abs_diff = 8e-3
        rel_diff = 8e-3

    cmp = testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff)

    assert cmp(musa.values().cpu(), cpu.values())
