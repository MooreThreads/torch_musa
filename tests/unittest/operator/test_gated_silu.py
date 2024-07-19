"""Test gated_silu operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,invalid-name,not-callable
import torch
import torch.nn.functional as F
import pytest
import torch_musa

from torch_musa import testing


def ref_gated_silu(input):  # pylint: disable=W0622
    x1, x2 = torch.chunk(input, 2, dim=-1)
    output = F.silu(x1) * x2
    return output


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("embedding_dim", [128, 512, 768])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("sequence_length", [1, 32, 128])
def test_gated_silu(embedding_dim, batch, sequence_length):
    input_shape = (batch, sequence_length, embedding_dim * 2)
    input_data = {
        "input": torch.randn(input_shape, dtype=torch.float16).to(torch.float32),
    }
    test = testing.OpTest(
        func=torch.gated_silu,
        refer_func=ref_gated_silu,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test._comparators = [testing.DefaultComparator(abs_diff=1e-2)]
    test.check_musafp16_vs_musafp32()
