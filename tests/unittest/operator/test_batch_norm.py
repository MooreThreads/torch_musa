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

train = [True, False]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("train", train)
def test_batch_norm(input_data, train):
    m = torch.nn.BatchNorm2d(100)
    m.train(train)
    output = m(input_data)
    output_musa = m.to("musa")(input_data.to("musa"))
    assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="fp16 batch_norm supported in QY2 or later",
)
@pytest.mark.parametrize("train", train)
def test_batch_norm_fp16(input_data, train):
    m = torch.nn.BatchNorm2d(100)
    m.train(train)
    output = m(input_data)
    m.half()
    input_data = input_data.half()
    output_musa = m.to("musa")(input_data.to("musa"))
    assert testing.DefaultComparator(abs_diff=1e-2)(output, output_musa.cpu().float())

input_data = [
     torch.randn(4, 100, 4, 4),
     torch.randn(8, 100, 8, 8),
     torch.randn(16, 100, 16, 16),
     torch.randn(64, 100, 16, 16)
 ]
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("train", [True])
def test_batch_norm_bwd(input_data, train):
    model = torch.nn.BatchNorm2d(100)
    musa_model = torch.nn.BatchNorm2d(100).to('musa')
    model.train(train)
    musa_model.train(train)
    output = model(input_data)
    output_musa = musa_model(input_data.to("musa"))
    output.sum().backward()
    output_musa.sum().backward()
    assert testing.DefaultComparator(abs_diff=1e-3)(model.weight.grad, musa_model.weight.grad.cpu())
    assert testing.DefaultComparator(abs_diff=1e-3)(model.bias.grad, musa_model.bias.grad.cpu())
