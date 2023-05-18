"""Test weight_norm operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, not-callable
import torch
import pytest
from torch.nn.utils import weight_norm
import torch_musa
from torch_musa import testing

data_type = testing.get_all_support_types()

input_data = [
    torch.randn(2, 20),
    torch.randn(4, 20),
    torch.randn(8, 20),
    torch.randn(16, 20),
    torch.randn(32, 20),
    torch.randn(64, 20),
    torch.randn(128, 20)
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_weight_norm(input_data):
    m = weight_norm(torch.nn.Linear(20, 40), name='weight')
    r_cpu = m(input_data)
    r_musa = m.to("musa")(input_data.to("musa"))
    testing.DefaultComparator()(r_cpu, r_musa.cpu())
    