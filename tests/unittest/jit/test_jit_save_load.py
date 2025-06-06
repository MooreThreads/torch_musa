"""Test jit features"""

# pylint: disable=redefined-builtin, unused-import, not-callable
import pytest
import torch
from torch import nn
import torch.nn.functional as F
import torch_musa
from torch_musa import testing


class MnistNet(nn.Module):
    """Define MnistNet model"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test_save_load_on_musa():
    """Testing save/load on musa"""
    net = MnistNet().to("musa")
    input = torch.randn(5, 1, 28, 28, device="musa")

    eager_output = net(input)

    traced_net = torch.jit.trace(net, [input], check_trace=False)
    traced_output = traced_net(input)

    assert (eager_output == traced_output).all()

    torch.jit.save(traced_net, "mnistnet.pt")
    loaded_net = torch.jit.load("mnistnet.pt").to("musa")
    loaded_output = loaded_net(input)

    assert (eager_output == loaded_output).all()


def test_save_cpu_load_musa():
    """Testing save on cpu and load to musa"""
    net = MnistNet()
    input = torch.randn(5, 1, 28, 28)
    eager_output = net(input).to("musa")

    traced_net = torch.jit.trace(net, [input], check_trace=False)
    with pytest.raises(
        RuntimeError,
        match="Expected weight tensor and input tensor to be on the same MUSA device, "
        "but got weight on cpu and input on musa:\\d",
    ):
        traced_net(input.to("musa"))

    traced_output = traced_net.to("musa")(input.to("musa"))

    comparator = testing.DefaultComparator()
    assert comparator(eager_output, traced_output)

    torch.jit.save(traced_net.to("cpu"), "mnistnet.pt")
    loaded_net = torch.jit.load("mnistnet.pt").to("musa")
    loaded_output = loaded_net(input.to("musa"))

    assert comparator(eager_output, loaded_output)


def test_save_musa_load_cpu():
    """Testing save on musa and load to cpu"""

    net = MnistNet().to("musa")
    input = torch.randn(5, 1, 28, 28, device="musa")

    eager_output = net(input)

    traced_net = torch.jit.trace(net, [input], check_trace=False)

    torch.jit.save(traced_net, "mnistnet.pt")
    loaded_net = torch.jit.load("mnistnet.pt").to("cpu")
    loaded_output = loaded_net(input.to("cpu")).to("musa")

    comparator = testing.DefaultComparator()
    assert comparator(eager_output, loaded_output)
