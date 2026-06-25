"""Test NestedTensor shape / structure operators and factory ops."""

# pylint: disable=C0115, C0116, W0611, C0103, C0411
import pytest
import torch
import torch_musa  # noqa: F401

from conftest import (
    DEVICE,
    DTYPE,
    DEFAULT_SHAPES,
    make_nt_pair,
    assert_nt_close,
)


# ========================= Shape Manipulation =========================


class TestNestedShape:

    def test_clone(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(nt_cpu.clone(), nt_musa.clone())

    def test_view(self):
        shapes = [(6,), (10,)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        assert_nt_close(nt_cpu.view(2, -1, 1), nt_musa.view(2, -1, 1))

    def test_transpose(self):
        shapes = [(2, 3), (4, 5)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        assert_nt_close(nt_cpu.transpose(1, 2), nt_musa.transpose(1, 2))

    def test_squeeze(self):
        shapes = [(1, 3), (1, 5)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        assert_nt_close(nt_cpu.squeeze(1), nt_musa.squeeze(1))

    def test_unsqueeze(self):
        shapes = [(3,), (5,)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        assert_nt_close(nt_cpu.unsqueeze(-1), nt_musa.unsqueeze(-1))

    def test_select(self):
        nt_cpu, nt_musa = make_nt_pair()
        cpu_t = nt_cpu.select(0, 0)
        musa_t = nt_musa.select(0, 0)
        torch.testing.assert_close(cpu_t, musa_t.cpu())

    def test_unbind(self):
        nt_cpu, nt_musa = make_nt_pair()
        cpu_parts = nt_cpu.unbind()
        musa_parts = nt_musa.unbind()
        assert len(cpu_parts) == len(musa_parts)
        for c, m in zip(cpu_parts, musa_parts):
            torch.testing.assert_close(c, m.cpu())

    def test_split_with_sizes(self):
        shapes = [(6, 4), (8, 4)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        cpu_parts = nt_cpu.split_with_sizes([2, 2], dim=-1)
        musa_parts = nt_musa.split_with_sizes([2, 2], dim=-1)
        assert len(cpu_parts) == len(musa_parts)
        for c, m in zip(cpu_parts, musa_parts):
            assert_nt_close(c, m)

    def test_chunk(self):
        shapes = [(6, 4), (8, 4)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        cpu_parts = nt_cpu.chunk(2, dim=-1)
        musa_parts = nt_musa.chunk(2, dim=-1)
        assert len(cpu_parts) == len(musa_parts)
        for c, m in zip(cpu_parts, musa_parts):
            assert_nt_close(c, m)

    def test_narrow(self):
        shapes = [(6, 4), (8, 4), (10, 4), (16, 16)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        assert_nt_close(nt_cpu.narrow(0, 1, 2), nt_musa.narrow(0, 1, 2))

    def test_cat(self):
        shapes = [(2, 3), (4, 3)]
        nt1_cpu, nt1_musa = make_nt_pair(shapes)
        nt2_cpu, nt2_musa = make_nt_pair(shapes)
        assert_nt_close(
            torch.cat([nt1_cpu, nt2_cpu], dim=0),
            torch.cat([nt1_musa, nt2_musa], dim=0),
        )

    def test_values(self):
        nt_cpu, nt_musa = make_nt_pair()
        torch.testing.assert_close(nt_cpu.values(), nt_musa.values().cpu())


# ========================= Factory / Copy Ops =========================


class TestNestedFactory:

    def test_empty_like(self):
        nt_cpu, nt_musa = make_nt_pair()
        result = torch.empty_like(nt_musa)
        assert result.is_nested
        for c, m in zip(nt_cpu.unbind(), result.unbind()):
            assert c.shape == m.shape

    def test_ones_like(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(torch.ones_like(nt_cpu), torch.ones_like(nt_musa))

    def test_fill_scalar(self):
        nt_cpu, nt_musa = make_nt_pair()
        nt_cpu.fill_(3.14)
        nt_musa.fill_(3.14)
        assert_nt_close(nt_cpu, nt_musa)

    def test_zero_(self):
        nt_cpu, nt_musa = make_nt_pair()
        nt_cpu.zero_()
        nt_musa.zero_()
        assert_nt_close(nt_cpu, nt_musa)

    def test_copy_(self):
        nt_cpu, nt_musa = make_nt_pair()
        nt2_cpu, nt2_musa = make_nt_pair()
        nt_cpu.copy_(nt2_cpu)
        nt_musa.copy_(nt2_musa)
        assert_nt_close(nt_cpu, nt_musa)

    def test_to_copy(self):
        _, nt_musa = make_nt_pair()
        result = nt_musa.to(torch.float64)
        assert result.dtype == torch.float64
        assert len(result.unbind()) == len(DEFAULT_SHAPES)

    def test_detach(self):
        nt_cpu, nt_musa = make_nt_pair(requires_grad=True)
        d_cpu = nt_cpu.detach()
        d_musa = nt_musa.detach()
        assert not d_cpu.requires_grad
        assert not d_musa.requires_grad
        assert_nt_close(d_cpu, d_musa)

    def test_alias(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(
            torch.ops.aten.alias(nt_cpu),
            torch.ops.aten.alias(nt_musa),
        )


# ========================= Metadata =========================


class TestNestedMetadata:

    def test_nested_tensor_size(self):
        nt_cpu, nt_musa = make_nt_pair()
        torch.testing.assert_close(
            nt_cpu._nested_tensor_size(), nt_musa._nested_tensor_size().cpu()
        )

    def test_nested_tensor_strides(self):
        nt_cpu, nt_musa = make_nt_pair()
        torch.testing.assert_close(
            nt_cpu._nested_tensor_strides(),
            nt_musa._nested_tensor_strides().cpu(),
        )

    def test_nested_tensor_storage_offsets(self):
        nt_cpu, nt_musa = make_nt_pair()
        torch.testing.assert_close(
            nt_cpu._nested_tensor_storage_offsets(),
            nt_musa._nested_tensor_storage_offsets().cpu(),
        )

    def test_is_same_size(self):
        nt1_cpu, nt1_musa = make_nt_pair()
        nt2_cpu, nt2_musa = make_nt_pair()
        assert nt1_cpu.is_same_size(nt2_cpu) == nt1_musa.is_same_size(nt2_musa)


# ========================= Padded Conversion =========================


class TestNestedPadded:

    def test_to_padded_tensor(self):
        nt_cpu, nt_musa = make_nt_pair()
        torch.testing.assert_close(
            nt_cpu.to_padded_tensor(0.0), nt_musa.to_padded_tensor(0.0).cpu()
        )

    def test_nested_from_padded_and_nested_example(self):
        nt_cpu, nt_musa = make_nt_pair()
        padded_cpu = nt_cpu.to_padded_tensor(0.0)
        padded_musa = padded_cpu.to(DEVICE)
        result_cpu = torch._nested_from_padded_and_nested_example(padded_cpu, nt_cpu)
        result_musa = torch._nested_from_padded_and_nested_example(padded_musa, nt_musa)
        assert_nt_close(result_cpu, result_musa)
