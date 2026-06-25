"""Test NestedTensor elementwise operators: unary, binary, comparison, activation, masking."""

# pylint: disable=C0115, C0116, W0611, C0411
import pytest
import torch
import torch_musa  # noqa: F401

from conftest import (
    DEVICE,
    DTYPE,
    make_nt_pair,
    make_binary_nt_pair,
    assert_nt_close,
    assert_nt_equal,
)


# ========================= Unary Ops =========================


class TestNestedUnary:

    def test_abs(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(torch.abs(nt_cpu), torch.abs(nt_musa))

    def test_abs_(self):
        nt_cpu, nt_musa = make_nt_pair()
        nt_cpu.abs_()
        nt_musa.abs_()
        assert_nt_close(nt_cpu, nt_musa)

    def test_neg(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(torch.neg(nt_cpu), torch.neg(nt_musa))

    def test_neg_(self):
        nt_cpu, nt_musa = make_nt_pair()
        nt_cpu.neg_()
        nt_musa.neg_()
        assert_nt_close(nt_cpu, nt_musa)

    def test_cos(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(torch.cos(nt_cpu), torch.cos(nt_musa))

    def test_sin(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(torch.sin(nt_cpu), torch.sin(nt_musa))

    def test_sgn(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(torch.sgn(nt_cpu), torch.sgn(nt_musa))

    def test_logical_not(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(torch.logical_not(nt_cpu), torch.logical_not(nt_musa))

    def test_isnan(self):
        tensors = [
            torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]]),
            torch.randn(4, 5),
        ]
        nt_cpu = torch.nested.nested_tensor([t.clone() for t in tensors])
        nt_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in tensors], device=DEVICE
        )
        assert_nt_equal(torch.isnan(nt_cpu), torch.isnan(nt_musa))

    def test_isinf(self):
        tensors = [
            torch.tensor([[1.0, float("inf"), 3.0], [float("-inf"), 5.0, 6.0]]),
            torch.randn(4, 5),
        ]
        nt_cpu = torch.nested.nested_tensor([t.clone() for t in tensors])
        nt_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in tensors], device=DEVICE
        )
        assert_nt_equal(torch.isinf(nt_cpu), torch.isinf(nt_musa))

    def test_isposinf(self):
        tensors = [
            torch.tensor([[1.0, float("inf"), 3.0], [float("-inf"), 5.0, 6.0]]),
            torch.randn(4, 5),
        ]
        nt_cpu = torch.nested.nested_tensor([t.clone() for t in tensors])
        nt_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in tensors], device=DEVICE
        )
        assert_nt_equal(torch.isposinf(nt_cpu), torch.isposinf(nt_musa))

    def test_isneginf(self):
        tensors = [
            torch.tensor([[1.0, float("inf"), 3.0], [float("-inf"), 5.0, 6.0]]),
            torch.randn(4, 5),
        ]
        nt_cpu = torch.nested.nested_tensor([t.clone() for t in tensors])
        nt_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in tensors], device=DEVICE
        )
        assert_nt_equal(torch.isneginf(nt_cpu), torch.isneginf(nt_musa))


# ========================= Activation Ops =========================


class TestNestedActivation:

    def test_relu(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(torch.relu(nt_cpu), torch.relu(nt_musa))

    def test_relu_(self):
        nt_cpu, nt_musa = make_nt_pair()
        torch.relu_(nt_cpu)
        torch.relu_(nt_musa)
        assert_nt_close(nt_cpu, nt_musa)

    def test_gelu(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(
            torch.nn.functional.gelu(nt_cpu),
            torch.nn.functional.gelu(nt_musa),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_silu(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(
            torch.nn.functional.silu(nt_cpu),
            torch.nn.functional.silu(nt_musa),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_tanh_(self):
        nt_cpu, nt_musa = make_nt_pair()
        nt_cpu.tanh_()
        nt_musa.tanh_()
        assert_nt_close(nt_cpu, nt_musa)


# ========================= Binary Ops =========================


class TestNestedBinary:

    def test_add_tensor(self):
        nt1_cpu, nt1_musa, nt2_cpu, nt2_musa = make_binary_nt_pair()
        assert_nt_close(nt1_cpu + nt2_cpu, nt1_musa + nt2_musa)

    def test_add__tensor(self):
        nt1_cpu, nt1_musa, nt2_cpu, nt2_musa = make_binary_nt_pair()
        nt1_cpu.add_(nt2_cpu)
        nt1_musa.add_(nt2_musa)
        assert_nt_close(nt1_cpu, nt1_musa)

    def test_sub_tensor(self):
        nt1_cpu, nt1_musa, nt2_cpu, nt2_musa = make_binary_nt_pair()
        assert_nt_close(nt1_cpu - nt2_cpu, nt1_musa - nt2_musa)

    def test_mul_tensor(self):
        nt1_cpu, nt1_musa, nt2_cpu, nt2_musa = make_binary_nt_pair()
        assert_nt_close(nt1_cpu * nt2_cpu, nt1_musa * nt2_musa)

    def test_mul__tensor(self):
        nt1_cpu, nt1_musa, nt2_cpu, nt2_musa = make_binary_nt_pair()
        nt1_cpu.mul_(nt2_cpu)
        nt1_musa.mul_(nt2_musa)
        assert_nt_close(nt1_cpu, nt1_musa)

    def test_div_tensor(self):
        nt1_cpu, nt1_musa, nt2_cpu, nt2_musa = make_binary_nt_pair()
        assert_nt_close(nt1_cpu / nt2_cpu, nt1_musa / nt2_musa, atol=1e-5, rtol=1e-5)

    def test_mul_scalar(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(nt_cpu * 2.5, nt_musa * 2.5)

    def test_div_scalar(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_close(nt_cpu / 2.5, nt_musa / 2.5, atol=1e-5, rtol=1e-5)


# ========================= Comparison Ops =========================


class TestNestedComparison:

    def test_eq_scalar(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_equal(nt_cpu.eq(0.0), nt_musa.eq(0.0))

    def test_gt_scalar(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_equal(nt_cpu.gt(0.0), nt_musa.gt(0.0))

    def test_ge_scalar(self):
        nt_cpu, nt_musa = make_nt_pair()
        assert_nt_equal(nt_cpu.ge(0.0), nt_musa.ge(0.0))

    def test_where(self):
        shapes = [(2, 3), (4, 5)]
        cond_tensors = [torch.randn(*s) > 0 for s in shapes]
        other_tensors = [torch.randn(*s) for s in shapes]

        cond_cpu = torch.nested.nested_tensor([c.clone().bool() for c in cond_tensors])
        cond_musa = torch.nested.nested_tensor(
            [c.clone().bool().to(DEVICE) for c in cond_tensors], device=DEVICE
        )

        other_cpu = torch.nested.nested_tensor([t.clone() for t in other_tensors])
        other_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in other_tensors], device=DEVICE
        )

        # self must be a non-nested tensor, other can be nested
        x = torch.tensor(1.0)

        assert_nt_close(
            torch.where(cond_cpu, x, other_cpu),
            torch.where(cond_musa, x.to(DEVICE), other_musa),
        )


# ========================= Masking Ops =========================


class TestNestedMask:

    def test_masked_fill_scalar(self):
        shapes = [(2, 3), (4, 5)]
        tensors = [torch.randn(*s) for s in shapes]
        masks = [torch.randn(*s) > 0 for s in shapes]

        nt_cpu = torch.nested.nested_tensor([t.clone() for t in tensors])
        mask_cpu = torch.nested.nested_tensor([m.clone() for m in masks])
        nt_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in tensors], device=DEVICE
        )
        mask_musa = torch.nested.nested_tensor(
            [m.clone().to(DEVICE) for m in masks], device=DEVICE
        )

        assert_nt_close(
            nt_cpu.masked_fill(mask_cpu, -1.0),
            nt_musa.masked_fill(mask_musa, -1.0),
        )
