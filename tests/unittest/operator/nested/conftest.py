"""Shared fixtures and helpers for NestedTensor tests."""

# pylint: disable=C0115, C0116, W0611, C0103
import pytest
import torch
import torch_musa  # noqa: F401

DEVICE = "musa"
DTYPE = torch.float32
DEFAULT_SHAPES = [(2, 3), (4, 5), (3, 7)]


def make_nt_pair(shapes=None, dtype=DTYPE, requires_grad=False):
    """Create a pair of identical nested tensors on CPU and MUSA."""
    if shapes is None:
        shapes = DEFAULT_SHAPES
    tensors = [torch.randn(*s, dtype=dtype) for s in shapes]
    for t in tensors:
        t.uniform_(-2, 2)
    nt_cpu = torch.nested.nested_tensor(
        [t.clone() for t in tensors], dtype=dtype, requires_grad=requires_grad
    )
    nt_musa = torch.nested.nested_tensor(
        [t.clone().to(DEVICE) for t in tensors],
        dtype=dtype,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    return nt_cpu, nt_musa


def make_binary_nt_pair(shapes=None, dtype=DTYPE):
    """Create two pairs of identical nested tensors for binary ops."""
    if shapes is None:
        shapes = DEFAULT_SHAPES
    t1 = [torch.randn(*s, dtype=dtype) for s in shapes]
    t2 = [torch.randn(*s, dtype=dtype) for s in shapes]
    for t in t1:
        t.uniform_(-2, 2)
    for t in t2:
        t.uniform_(-2, 2)
    nt1_cpu = torch.nested.nested_tensor([t.clone() for t in t1], dtype=dtype)
    nt1_musa = torch.nested.nested_tensor(
        [t.clone().to(DEVICE) for t in t1], dtype=dtype, device=DEVICE
    )
    nt2_cpu = torch.nested.nested_tensor([t.clone() for t in t2], dtype=dtype)
    nt2_musa = torch.nested.nested_tensor(
        [t.clone().to(DEVICE) for t in t2], dtype=dtype, device=DEVICE
    )
    return nt1_cpu, nt1_musa, nt2_cpu, nt2_musa


def assert_nt_close(nt_cpu, nt_musa, **kwargs):
    """Compare nested tensors by unbinding and checking each component."""
    cpu_parts = nt_cpu.unbind()
    musa_parts = nt_musa.unbind()
    assert len(cpu_parts) == len(musa_parts)
    for c, m in zip(cpu_parts, musa_parts):
        torch.testing.assert_close(c, m.cpu(), **kwargs)


def assert_nt_equal(nt_cpu, nt_musa):
    """Compare nested tensors for exact equality."""
    cpu_parts = nt_cpu.unbind()
    musa_parts = nt_musa.unbind()
    assert len(cpu_parts) == len(musa_parts)
    for c, m in zip(cpu_parts, musa_parts):
        assert torch.equal(c, m.cpu())
