""" pytest for lstm """

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, not-callable
import copy
import pytest

import torch
from torch import nn
import torch_musa
from torch_musa import testing


input_shapes = [
    (2, 5, 16),  # (seq_len, batch, input_size)
    (4, 8, 16),
    (10, 32, 16),
    (64, 4, 16),  # edge case: zero seq_len
]
dtypes = testing.get_float_types()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_lstm(input_shape, dtype):
    seq_len, batch, input_size = input_shape
    hidden_size = 32
    lstm = nn.LSTM(input_size, hidden_size, num_layers=1).to(dtype)
    input_tensor = torch.randn(seq_len, batch, input_size, dtype=dtype)

    output_cpu, (hn_cpu, cn_cpu) = lstm(input_tensor)

    lstm_musa = copy.deepcopy(lstm).to("musa")
    input_musa = input_tensor.to("musa")
    output_musa, (hn_musa, cn_musa) = lstm_musa(input_musa)

    comparator = testing.DefaultComparator()
    comparator(output_cpu, output_musa.cpu())
    comparator(hn_cpu, hn_musa.cpu())
    comparator(cn_cpu, cn_musa.cpu())
