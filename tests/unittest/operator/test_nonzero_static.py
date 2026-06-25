"""Tests for nonzero_static operators (public API)."""

# pylint: disable=missing-function-docstring, redefined-outer-name
import torch

from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_nonzero_static_basic():
    # 1D bool mask, simple case
    mask_cpu = torch.tensor([0, 1, 0, 2, 3], dtype=torch.int32) != 0
    size = mask_cpu.numel()

    out_cpu = torch.ops.aten.nonzero_static.default(mask_cpu, size=size, fill_value=-1)

    mask_musa = mask_cpu.to("musa")
    out_musa = torch.ops.aten.nonzero_static.default(
        mask_musa, size=size, fill_value=-1
    ).cpu()

    # CPU / MUSA 数值一致（拉平比较，忽略具体二维布局差异）
    assert out_musa.numel() == out_cpu.numel()
    assert out_musa.dtype == out_cpu.dtype
    comparator = testing.DefaultComparator(abs_diff=0, rel_diff=0)
    assert comparator(out_cpu.reshape(-1), out_musa.reshape(-1))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_nonzero_static_out():
    # also use 1D mask to keep semantics simple
    mask_cpu = torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.int32) != 0
    size = mask_cpu.numel()
    out_cpu = torch.empty(size, dtype=torch.int64)
    out_cpu_res = torch.ops.aten.nonzero_static.out(
        mask_cpu, size=size, fill_value=-1, out=out_cpu
    )

    mask_musa = mask_cpu.to("musa")
    out_musa = torch.empty(size, dtype=torch.int64, device="musa")
    out_musa_res = torch.ops.aten.nonzero_static.out(
        mask_musa, size=size, fill_value=-1, out=out_musa
    ).cpu()

    # CPU / MUSA 数值一致（同样拉平比较）
    assert out_musa_res.numel() == out_cpu_res.numel()
    assert out_musa_res.dtype == out_cpu_res.dtype
    comparator = testing.DefaultComparator(abs_diff=0, rel_diff=0)
    assert comparator(out_cpu_res.reshape(-1), out_musa_res.reshape(-1))
