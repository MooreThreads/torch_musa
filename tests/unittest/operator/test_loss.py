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

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_nll_loss2d():
    input_data = torch.randn(3,20,224,224, requires_grad=True)
    target_data = torch.randint(0,19,(3,224,224),dtype=torch.int64)
    loss = torch.nn.CrossEntropyLoss()
    output_cpu = loss(input_data, target_data)
    musa_data = torch.tensor(input_data.detach().numpy(), requires_grad=True, device="musa")
    musa_target = torch.tensor(target_data.detach().numpy(), device="musa")
    output_musa = loss(musa_data, musa_target)
    output_cpu.backward()
    output_musa.backward()
    assert pytest.approx(output_cpu.detach(), 1e-6) == output_musa.detach().to("cpu")
    assert pytest.approx(input_data.grad, 1e-6) == musa_data.grad.to("cpu")
