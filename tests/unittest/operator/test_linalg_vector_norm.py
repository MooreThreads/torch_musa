"""Test linaly_vector_norm operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,not-callable
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    torch.randn(4, 100, 4, 4),
    torch.randn(8, 100, 8, 8),
    torch.randn(16, 100, 16, 16),
]
dim = [0,1,2,3]
order = [1,3]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dim", dim)
@pytest.mark.parametrize("order", order)
def test_linalg_vector_norm(input_data,dim,order):
    m = torch.linalg.vector_norm
    output = m(input_data,dim,order)
    musa_input = input_data.to('musa')
    output_musa = m(musa_input,dim,order)
    assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa.cpu())
