"""Test jagged <-> padded dense forward ops on MUSA vs CPU."""

# pylint: disable=C0115, C0116, W0611, C0103
import pytest
import torch
import torch_musa  # noqa: F401


DEVICE = "musa"


def _make_offsets(lengths, device):
    offsets = [0]
    for l in lengths:
        offsets.append(offsets[-1] + int(l))
    return torch.tensor(offsets, device=device, dtype=torch.int64)


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("max_len", [16, 32])
def test_jagged_to_padded_dense_forward(B, D, max_len):
    lengths = [(i * 3) % (max_len + 1) for i in range(B)]
    total_L = sum(lengths)

    offsets_cpu = _make_offsets(lengths, device="cpu")
    offsets_musa = offsets_cpu.to(DEVICE)

    values_cpu = torch.randn((total_L, D), device="cpu", dtype=torch.float32)
    values_musa = values_cpu.to(DEVICE)

    padding_value = -1.5
    out_cpu = torch.ops.aten._jagged_to_padded_dense_forward(
        values_cpu, [offsets_cpu], [max_len], padding_value
    )
    out_musa = torch.ops.aten._jagged_to_padded_dense_forward(
        values_musa, [offsets_musa], [max_len], padding_value
    )

    torch.testing.assert_close(out_cpu, out_musa.cpu())

    assert list(out_musa.shape) == [B, max_len, D]
    if total_L == 0:
        assert torch.all(out_musa.cpu() == padding_value)


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("max_len", [16, 32])
def test_padded_dense_to_jagged_forward(B, D, max_len):
    lengths = [(i * 5) % (max_len + 1) for i in range(B)]
    total_L = sum(lengths)

    offsets_cpu = _make_offsets(lengths, device="cpu")
    offsets_musa = offsets_cpu.to(DEVICE)

    dense_cpu = torch.randn((B, max_len, D), device="cpu", dtype=torch.float32)
    dense_musa = dense_cpu.to(DEVICE)

    out_cpu = torch.ops.aten._padded_dense_to_jagged_forward(
        dense_cpu, [offsets_cpu], total_L
    )
    out_musa = torch.ops.aten._padded_dense_to_jagged_forward(
        dense_musa, [offsets_musa], total_L
    )

    assert list(out_cpu.shape) == [total_L, D]
    assert list(out_musa.shape) == [total_L, D]
    torch.testing.assert_close(out_cpu, out_musa.cpu())
