"""Test conv2d operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
             {
              'input': torch.randn(2, 3, 16, 16, requires_grad=True),
              'kernel_size': 3,
              'stride': 1,
              'padding': 1,
              'bias': False,
              'in_channels': 3,
              'out_channels': 1,
              'dilation': 1,
              'groups': 1
             },
             {
              'input': torch.randn(2, 4, 64, 64, requires_grad=True),
              'kernel_size': 5,
              'stride': 1,
              'padding': 1,
              'bias': False,
              'in_channels': 4,
              'out_channels': 1,
              'dilation': 1,
              'groups': 1
             }
]

def set_same_weight(model, other):
    for key in model.state_dict().keys():
        other.state_dict()[key].data.copy_(model.state_dict()[key].data)


@pytest.mark.parametrize('input_data', input_data)
def test_conv2d(input_data):
    """Test conv2d operators."""
    cpu_input = input_data['input']
    musa_input = input_data['input']
    conv2d = torch.nn.Conv2d(in_channels=input_data['in_channels'],
                                out_channels=input_data['out_channels'],
                                kernel_size=input_data['kernel_size'],
                                stride=input_data['stride'],
                                padding=input_data['padding'],
                                dilation=input_data['dilation'],
                                groups=input_data['groups'],
                                bias=input_data['bias'],
                                device="cpu")
    musa_conv2d = torch.nn.Conv2d(in_channels=input_data['in_channels'],
                                out_channels=input_data['out_channels'],
                                kernel_size=input_data['kernel_size'],
                                stride=input_data['stride'],
                                padding=input_data['padding'],
                                dilation=input_data['dilation'],
                                groups=input_data['groups'],
                                bias=input_data['bias'],
                                device="musa")
    set_same_weight(conv2d, musa_conv2d)
    cpu_output = conv2d(cpu_input)
    musa_output = musa_conv2d(musa_input.to("musa"))
    comparator = testing.DefaultComparator()
    assert comparator(cpu_output, musa_output.cpu())
    cpu_output.sum().backward()
    musa_output.sum().backward()
    assert comparator(cpu_input.grad, musa_input.grad.cpu())
