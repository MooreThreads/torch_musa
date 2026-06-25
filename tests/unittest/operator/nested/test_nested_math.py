"""Test NestedTensor math / neural-network operators, dropout, and backward."""

# pylint: disable=C0115, C0116, W0611, C0103, E1102, C0411
import pytest
import torch
import torch_musa  # noqa: F401

from conftest import (
    DEVICE,
    DTYPE,
    make_nt_pair,
    assert_nt_close,
)


# ========================= Matmul / Linear =========================


class TestNestedMath:

    def test_matmul(self):
        t1 = [torch.randn(2, 4), torch.randn(3, 4)]
        t2 = [torch.randn(4, 5), torch.randn(4, 5)]
        nt1_cpu = torch.nested.nested_tensor([t.clone() for t in t1])
        nt2_cpu = torch.nested.nested_tensor([t.clone() for t in t2])
        nt1_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in t1], device=DEVICE
        )
        nt2_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in t2], device=DEVICE
        )
        assert_nt_close(
            torch.matmul(nt1_cpu, nt2_cpu),
            torch.matmul(nt1_musa, nt2_musa),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_linear(self):
        weight = torch.randn(8, 3)
        bias = torch.randn(8)
        shapes = [(2, 3), (4, 3)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        assert_nt_close(
            torch.nn.functional.linear(nt_cpu, weight, bias),
            torch.nn.functional.linear(nt_musa, weight.to(DEVICE), bias.to(DEVICE)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_layer_norm(self):
        shapes = [(2, 4), (3, 4)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        ln = torch.nn.LayerNorm(4)
        ln_musa = torch.nn.LayerNorm(4).to(DEVICE)
        ln_musa.load_state_dict(ln.state_dict())
        assert_nt_close(ln(nt_cpu), ln_musa(nt_musa), atol=1e-5, rtol=1e-5)

    def test_embedding(self):
        emb = torch.nn.Embedding(10, 4)
        emb_musa = torch.nn.Embedding(10, 4).to(DEVICE)
        emb_musa.load_state_dict(emb.state_dict())

        idx = [torch.randint(0, 10, (3,)), torch.randint(0, 10, (5,))]
        nt_cpu = torch.nested.nested_tensor([t.clone() for t in idx])
        nt_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in idx], device=DEVICE
        )
        assert_nt_close(emb(nt_cpu), emb_musa(nt_musa), atol=1e-5, rtol=1e-5)

    def test_all_dim(self):
        shapes = [(2, 3), (4, 5)]
        tensors = [(torch.randn(*s) > 0) for s in shapes]
        nt_cpu = torch.nested.nested_tensor([t.clone() for t in tensors])
        nt_musa = torch.nested.nested_tensor(
            [t.clone().to(DEVICE) for t in tensors], device=DEVICE
        )
        assert_nt_close(nt_cpu.all(-1), nt_musa.all(-1))

    def test_safe_softmax(self):
        shapes = [(2, 3), (4, 5)]
        nt_cpu, nt_musa = make_nt_pair(shapes)
        assert_nt_close(
            torch.ops.aten._safe_softmax(nt_cpu, -1),
            torch.ops.aten._safe_softmax(nt_musa, -1),
            atol=1e-5,
            rtol=1e-5,
        )


# ========================= Dropout =========================


class TestNestedDropout:

    def test_native_dropout_eval(self):
        nt_cpu, nt_musa = make_nt_pair()
        result_cpu, _ = torch.native_dropout(nt_cpu, 0.0, False)
        result_musa, _ = torch.native_dropout(nt_musa, 0.0, False)
        assert_nt_close(result_cpu, result_musa)


# ========================= Backward Ops =========================


class TestNestedBackward:

    def test_gelu_backward(self):
        nt_cpu, nt_musa = make_nt_pair(requires_grad=True)
        out_cpu = torch.nn.functional.gelu(nt_cpu)
        out_musa = torch.nn.functional.gelu(nt_musa)

        grad_cpu = torch.nested.nested_tensor(
            [torch.ones_like(t) for t in out_cpu.unbind()]
        )
        grad_musa = torch.nested.nested_tensor(
            [torch.ones_like(t) for t in out_musa.unbind()], device=DEVICE
        )
        out_cpu.backward(grad_cpu)
        out_musa.backward(grad_musa)
        assert_nt_close(nt_cpu.grad, nt_musa.grad, atol=1e-5, rtol=1e-5)

    def test_silu_backward(self):
        nt_cpu, nt_musa = make_nt_pair(requires_grad=True)
        out_cpu = torch.nn.functional.silu(nt_cpu)
        out_musa = torch.nn.functional.silu(nt_musa)

        grad_cpu = torch.nested.nested_tensor(
            [torch.ones_like(t) for t in out_cpu.unbind()]
        )
        grad_musa = torch.nested.nested_tensor(
            [torch.ones_like(t) for t in out_musa.unbind()], device=DEVICE
        )
        out_cpu.backward(grad_cpu)
        out_musa.backward(grad_musa)
        assert_nt_close(nt_cpu.grad, nt_musa.grad, atol=1e-5, rtol=1e-5)

    def test_relu_backward(self):
        nt_cpu, nt_musa = make_nt_pair(requires_grad=True)
        out_cpu = torch.relu(nt_cpu)
        out_musa = torch.relu(nt_musa)

        grad_cpu = torch.nested.nested_tensor(
            [torch.ones_like(t) for t in out_cpu.unbind()]
        )
        grad_musa = torch.nested.nested_tensor(
            [torch.ones_like(t) for t in out_musa.unbind()], device=DEVICE
        )
        out_cpu.backward(grad_cpu)
        out_musa.backward(grad_musa)
        assert_nt_close(nt_cpu.grad, nt_musa.grad, atol=1e-5, rtol=1e-5)
