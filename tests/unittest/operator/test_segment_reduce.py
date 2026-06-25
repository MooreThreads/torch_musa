"""Test segment_reduce operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name
import torch
import pytest

from torch_musa import testing


# 仅测试 axis=0 的情况，保证 lengths 最后一维与 axis 一致，避免 API 约束报错
segment_configs = [
    # simple 1D, three segments, sum / mean / min / max / prod
    {
        "shape": (6,),
        "lengths": [2, 2, 2],
        "axis": 0,
    },
    # 2D, reduce on axis=0
    {
        "shape": (4, 5),
        "lengths": [1, 1, 1, 1],
        "axis": 0,
    },
    # include empty segment (only for reductions with well-defined NAN behaviour)
    {
        "shape": (4,),
        "lengths": [0, 2, 2],
        "axis": 0,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("cfg", segment_configs)
@pytest.mark.parametrize("reduction", ["sum", "mean", "min", "max", "prod"])
def test_segment_reduce_lengths_forward(cfg, reduction):
    # prepare input
    data = torch.randn(cfg["shape"], dtype=torch.float32)
    lengths = torch.tensor(cfg["lengths"], dtype=torch.int64)
    axis = cfg["axis"]

    # CPU reference
    cpu_out = torch.segment_reduce(
        data, reduce=reduction, lengths=lengths, axis=axis, initial=None
    )

    # MUSA result
    musa_data = data.to("musa")
    musa_lengths = lengths.to("musa")
    musa_out = torch.segment_reduce(
        musa_data, reduce=reduction, lengths=musa_lengths, axis=axis, initial=None
    ).cpu()

    # 对包含空 segment + mean 的情况，CPU/MUSA 都会产生 NaN，需 equal_nan=True
    if reduction == "mean" and any(l == 0 for l in cfg["lengths"]):
        comparator = testing.DefaultComparator(
            abs_diff=1e-5, rel_diff=1e-4, equal_nan=True
        )
    else:
        comparator = testing.DefaultComparator(abs_diff=1e-5, rel_diff=1e-4)
    assert comparator(cpu_out, musa_out)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("cfg", [segment_configs[0], segment_configs[2]])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_segment_reduce_offsets_forward(cfg, reduction):
    # build offsets from lengths
    lengths = torch.tensor(cfg["lengths"], dtype=torch.int64)
    axis = cfg["axis"]

    if axis != 0:
        pytest.skip("offsets test focuses on 1D / axis=0 for simplicity")

    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64), lengths.cumsum(dim=0)], dim=0
    )

    total_len = int(lengths.sum().item())
    data = torch.randn((total_len,), dtype=torch.float32)

    cpu_out = torch.segment_reduce(
        data, reduce=reduction, offsets=offsets, axis=axis, initial=None
    )

    musa_data = data.to("musa")
    musa_offsets = offsets.to("musa")
    musa_out = torch.segment_reduce(
        musa_data, reduce=reduction, offsets=musa_offsets, axis=axis, initial=None
    ).cpu()

    if reduction == "mean" and any(l == 0 for l in cfg["lengths"]):
        comparator = testing.DefaultComparator(
            abs_diff=1e-5, rel_diff=1e-4, equal_nan=True
        )
    else:
        comparator = testing.DefaultComparator(abs_diff=1e-5, rel_diff=1e-4)
    assert comparator(cpu_out, musa_out)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_segment_reduce_backward_lengths(reduction):
    # simple shape with non-trivial gradients (axis=0 to satisfy API constraints)
    shape = (4, 5)
    lengths = torch.tensor([1, 2, 1, 0], dtype=torch.int64)
    axis = 0

    data_cpu = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    out_cpu = torch.segment_reduce(
        data_cpu, reduce=reduction, lengths=lengths, axis=axis, initial=None
    )
    loss_cpu = out_cpu.sum()
    loss_cpu.backward()
    grad_cpu = data_cpu.grad.clone()

    data_musa = data_cpu.detach().clone().to("musa").requires_grad_(True)
    lengths_musa = lengths.to("musa")
    out_musa = torch.segment_reduce(
        data_musa, reduce=reduction, lengths=lengths_musa, axis=axis, initial=None
    )
    loss_musa = out_musa.sum()
    loss_musa.backward()
    grad_musa = data_musa.grad.cpu()

    comparator = testing.DefaultComparator(abs_diff=1e-4, rel_diff=1e-3, equal_nan=True)
    assert comparator(grad_cpu, grad_musa)
