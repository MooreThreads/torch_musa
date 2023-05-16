"""Test batch_norm operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,not-callable
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    torch.randn(4, 100, 4, 4),
    torch.randn(8, 100, 8, 8),
    torch.randn(16, 100, 16, 16),
    torch.randn(64, 100, 16, 16),
    torch.randn(256, 100, 16, 16),
]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_batch_norm(input_data):
    m = torch.nn.BatchNorm2d(100)
    output = m(input_data)
    output_musa = m.to('musa')(input_data.to('musa'))
    assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa.cpu())
