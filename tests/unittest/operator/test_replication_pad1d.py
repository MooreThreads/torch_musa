"""Test reflection_pad operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from functools import partial
import torch
from torch import nn
import pytest

from torch_musa import testing


input_data = [
    torch.arange(8, dtype=torch.float).reshape(1, 2, 4),
    torch.arange(200, dtype=torch.float).reshape(2, 10, 10),
    torch.rand(0, 10, 10),
]

# not support for fp16 and int
support_dtypes = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_replication_pad1d(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReplicationPad1d(2)
    output_cpu = m(input_data)
    output_musa = m(input_data.to("musa"))
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")
