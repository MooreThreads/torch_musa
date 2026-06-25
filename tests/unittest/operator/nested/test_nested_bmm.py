"""Test NestedTensor bmm on MUSA vs CPU."""

# pylint: disable=C0115, C0116, W0611, C0103, E1102, C0411
import pytest
import torch
import torch_musa  # noqa: F401

from conftest import DEVICE, DTYPE, make_nt_pair, assert_nt_close


class TestNestedBmm:
    def test_bmm_nested_tensor_inputs(self):
        # Both inputs are NestedTensor
        shapes1 = [(16, 128), (128, 128)]
        shapes2 = [(128, 128), (128, 128)]
        nt1_cpu, nt1_musa = make_nt_pair(shapes=shapes1, dtype=DTYPE)
        nt2_cpu, nt2_musa = make_nt_pair(shapes=shapes2, dtype=DTYPE)

        out_cpu = torch.bmm(nt1_cpu, nt2_cpu)
        out_musa = torch.bmm(nt1_musa, nt2_musa)

        assert_nt_close(out_cpu, out_musa, atol=1e-5, rtol=1e-5)

    def test_bmm_mixed_nested_dense(self):
        # One input NestedTensor, one input dense 3D tensor
        # Nested shapes: (M0, K), (M1, K) with varying M, shared K
        shapes = [(16, 128), (128, 128)]
        nt_cpu, nt_musa = make_nt_pair(shapes=shapes, dtype=DTYPE)

        # Dense batch: all batches must have the same 2D shape (K, N)
        t2 = [torch.randn(128, 256, dtype=DTYPE)] * 2
        mat2_cpu = torch.stack(t2, dim=0)
        mat2_musa = mat2_cpu.to(DEVICE)

        out_cpu = torch.bmm(nt_cpu, mat2_cpu)
        out_musa = torch.bmm(nt_musa, mat2_musa)

        assert_nt_close(out_cpu, out_musa, atol=1e-5, rtol=1e-5)
