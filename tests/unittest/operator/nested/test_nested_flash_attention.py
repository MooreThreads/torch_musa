"""Test NestedTensor flash attention operators on MUSA."""

# pylint: disable=C0115, C0116, W0611, C0103, C0411
import pytest
import torch
import torch_musa  # noqa: F401

from conftest import DEVICE
from torch_musa import testing


def _make_base_qkv(dtype):
    q_list = [
        torch.randn(4, 128, 128, dtype=dtype),
        torch.randn(4, 256, 128, dtype=dtype),
    ]
    k_list = [
        torch.randn(4, 128, 128, dtype=dtype),
        torch.randn(4, 256, 128, dtype=dtype),
    ]
    v_list = [
        torch.randn(4, 128, 128, dtype=dtype),
        torch.randn(4, 256, 128, dtype=dtype),
    ]
    return q_list, k_list, v_list


def _make_nested_qkv(base_lists, device, dtype, requires_grad=False):
    q_base, k_base, v_base = base_lists
    q_list = [
        t.detach().clone().to(device=device, dtype=dtype).requires_grad_(requires_grad)
        for t in q_base
    ]
    k_list = [
        t.detach().clone().to(device=device, dtype=dtype).requires_grad_(requires_grad)
        for t in k_base
    ]
    v_list = [
        t.detach().clone().to(device=device, dtype=dtype).requires_grad_(requires_grad)
        for t in v_base
    ]
    query = torch.nested.nested_tensor(
        q_list, device=device, dtype=dtype, requires_grad=requires_grad
    )
    key = torch.nested.nested_tensor(
        k_list, device=device, dtype=dtype, requires_grad=requires_grad
    )
    value = torch.nested.nested_tensor(
        v_list, device=device, dtype=dtype, requires_grad=requires_grad
    )
    return query, key, value, q_list, k_list, v_list


def _assert_nested_close(ref_nt, result_nt, *, atol, rtol):
    ref_parts = ref_nt.unbind()
    result_parts = result_nt.unbind()
    assert len(ref_parts) == len(result_parts)
    for ref, result in zip(ref_parts, result_parts):
        torch.testing.assert_close(
            ref, result.detach().cpu().to(ref.dtype), atol=atol, rtol=rtol
        )


def _reference_forward(q_list, k_list, v_list):
    outputs = []
    for q, k, v in zip(q_list, k_list, v_list):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), dropout_p=0.0
        )
        outputs.append(out.squeeze(0))
    return torch.nested.nested_tensor(outputs, dtype=q_list[0].dtype)


def _reference_backward(q_list, k_list, v_list, grad_out_list):
    dq_list = []
    dk_list = []
    dv_list = []
    for q, k, v, grad_out in zip(q_list, k_list, v_list, grad_out_list):
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_ref.unsqueeze(0), k_ref.unsqueeze(0), v_ref.unsqueeze(0), dropout_p=0.0
        )
        out.backward(grad_out.unsqueeze(0))
        dq_list.append(q_ref.grad.detach())
        dk_list.append(k_ref.grad.detach())
        dv_list.append(v_ref.grad.detach())
    return (
        torch.nested.nested_tensor(dq_list, dtype=q_list[0].dtype),
        torch.nested.nested_tensor(dk_list, dtype=k_list[0].dtype),
        torch.nested.nested_tensor(dv_list, dtype=v_list[0].dtype),
    )


@pytest.mark.skipif(
    testing.get_musa_arch() < 31,
    reason="Flash attention varlen (nested tensors) is not supported on mp22, "
    "requires MUSA arch >= 31 (mp31).",
)
class TestNestedFlashAttention:
    def test_scaled_dot_product_flash_attention_forward_nested(self):
        base_lists = _make_base_qkv(torch.half)
        query_musa, key_musa, value_musa, _, _, _ = _make_nested_qkv(
            base_lists, DEVICE, torch.half
        )
        _, _, _, q_list_cpu, k_list_cpu, v_list_cpu = _make_nested_qkv(
            base_lists, "cpu", torch.float32
        )
        expected = _reference_forward(q_list_cpu, k_list_cpu, v_list_cpu)
        output, _, cum_seq_q, cum_seq_k, max_q, max_k, _, _, debug_mask = (
            torch.ops.aten._scaled_dot_product_flash_attention(
                query_musa,
                key_musa,
                value_musa,
                0.0,
                False,
                False,
            )
        )

        _assert_nested_close(expected, output, atol=5e-2, rtol=1e-3)
        torch.testing.assert_close(
            cum_seq_q.cpu(), torch.tensor([0, 128, 384], dtype=cum_seq_q.dtype)
        )
        torch.testing.assert_close(
            cum_seq_k.cpu(), torch.tensor([0, 128, 384], dtype=cum_seq_k.dtype)
        )
        assert int(max_q) == 256
        assert int(max_k) == 256
        assert debug_mask.numel() == 0

    def test_scaled_dot_product_flash_attention_backward_nested(self):
        base_lists = _make_base_qkv(torch.half)
        query_musa, key_musa, value_musa, _, _, _ = _make_nested_qkv(
            base_lists, DEVICE, torch.half
        )
        _, _, _, q_list_cpu, k_list_cpu, v_list_cpu = _make_nested_qkv(
            base_lists, "cpu", torch.float32
        )

        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            _,
        ) = torch.ops.aten._scaled_dot_product_flash_attention(
            query_musa,
            key_musa,
            value_musa,
            0.0,
            False,
            False,
        )

        grad_out_list_cpu = [torch.randn_like(t) for t in q_list_cpu]
        grad_out_musa = torch.nested.nested_tensor(
            [t.to(DEVICE, dtype=torch.half) for t in grad_out_list_cpu],
            device=DEVICE,
            dtype=torch.half,
        )
        grad_q, grad_k, grad_v = (
            torch.ops.aten._scaled_dot_product_flash_attention_backward(
                grad_out_musa,
                query_musa,
                key_musa,
                value_musa,
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                0.0,
                False,
                philox_seed,
                philox_offset,
            )
        )

        expected_dq, expected_dk, expected_dv = _reference_backward(
            q_list_cpu, k_list_cpu, v_list_cpu, grad_out_list_cpu
        )
        _assert_nested_close(expected_dq, grad_q, atol=5e-2, rtol=1e-3)
        _assert_nested_close(expected_dk, grad_k, atol=5e-2, rtol=1e-3)
        _assert_nested_close(expected_dv, grad_v, atol=5e-2, rtol=1e-3)
