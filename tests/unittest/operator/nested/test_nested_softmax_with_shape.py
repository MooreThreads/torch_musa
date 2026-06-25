"""Test _nested_tensor_softmax_with_shape on MUSA vs CPU."""

# pylint: disable=C0115, C0116, W0611, C0103, E1102, C0411
import torch
from torch_musa import testing

from conftest import DEVICE, DTYPE, make_nt_pair


class TestNestedSoftmaxWithShape:
    def test_nested_tensor_softmax_with_shape_basic(self):
        shapes = [(128, 128), (256, 128), (384, 128)]
        query_cpu, query_musa = make_nt_pair(shapes=shapes, dtype=DTYPE)

        batch_size = len(shapes)
        max_seq_len = max(s[0] for s in shapes)
        num_heads = 4

        # Attention scores tensor shaped as [B, H, T_max, T_max]
        scores_cpu = torch.randn(
            batch_size,
            num_heads,
            max_seq_len,
            max_seq_len,
            dtype=DTYPE,
        )
        scores_musa = scores_cpu.to(DEVICE)

        out_cpu = torch._nested_tensor_softmax_with_shape(scores_cpu, query_cpu)
        out_musa = torch._nested_tensor_softmax_with_shape(
            scores_musa, query_musa
        ).cpu()
        for i, shape in enumerate(shapes):
            out_musa[i, :, shape[0] :, :].zero_()
        assert out_musa.shape == out_cpu.shape
        assert out_musa.dtype == out_cpu.dtype

        cmp = testing.DefaultComparator(abs_diff=1e-6, rel_diff=1e-6)
        assert cmp(out_musa, out_cpu)
