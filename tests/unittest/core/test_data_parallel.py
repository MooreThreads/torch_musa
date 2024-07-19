"""Demo test of DistributedDataParall"""

# pylint: disable = W0611, C0103, C0116
import os
import torch
import pytest
from torch import nn
from torch import optim
import torch_musa
from torch_musa import testing


class Model(nn.Module):
    """
    Toy model.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 3, 2, 1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1280, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.linear(x)


def train(num_gpus):
    model = Model().to("musa")
    num_gpus = list(range(num_gpus))
    model = nn.DataParallel(model, device_ids=num_gpus)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    for _ in range(5):
        input_tensor = torch.randn(
            1, 3, 32, 32, dtype=torch.float, requires_grad=True
        ).to("musa")
        target_tensor = torch.zeros(10, dtype=torch.float).to("musa")
        output_tensor = model(input_tensor)
        loss_f = nn.MSELoss()
        loss = loss_f(output_tensor, target_tensor)
        loss.backward()
        optimizer.step()


@testing.skip_if_not_multiple_musa_device
def test_DataParallel():
    train(torch.musa.device_count())


if __name__ == "__main__":
    train(2)
