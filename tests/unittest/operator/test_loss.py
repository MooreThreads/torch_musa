"""Test loss operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = testing.get_raw_data()
# dtype of input tensor of mse_loss only support Float32 in muDNN now.
support_dtypes = [torch.float32]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_mse_loss(input_data, dtype):
    input_data = input_data.to(dtype)
    target_data = torch.rand_like(input_data)
    loss = torch.nn.MSELoss()
    output_cpu = loss(input_data, target_data)
    output_musa = loss(input_data.to("musa"), target_data.to("musa"))
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")
