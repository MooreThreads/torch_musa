# pylint: disable= missing-module-docstring, missing-class-docstring,missing-function-docstring,unused-import,unused-variable,not-callable
import os
import random
import pytest
import torch
from torch import nn
import numpy as np
import torch_musa
from torch_musa import testing


def set_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)
    torch.musa.manual_seed_all(seed)


# simple linear model to test the usability of amp .
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def __call__(self, x):
        return self.forward(x)


DEVICE = "musa"


def train_in_amp(low_dtype=torch.float16):
    set_seed()
    model = SimpleModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # create the scaler object
    scaler = torch.musa.amp.GradScaler()

    inputs = torch.randn(6, 5).to(DEVICE)  # 将数据移至GPU
    targets = torch.randn(6, 3).to(DEVICE)
    for step in range(20):
        optimizer.zero_grad()
        # create autocast environment
        with torch.musa.amp.autocast(dtype=low_dtype):
            outputs = model(inputs)
            assert outputs.dtype == low_dtype
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return loss


def train_in_fp32():
    set_seed()
    model = SimpleModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    inputs = torch.randn(6, 5).to(DEVICE)  # 将数据移至GPU
    targets = torch.randn(6, 3).to(DEVICE)
    for step in range(20):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    return loss


@pytest.mark.parametrize("low_dtype", [torch.float16])
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_amp_autocast_fp16(low_dtype):
    res_fp32 = train_in_fp32()
    res_amp = train_in_amp(low_dtype=low_dtype)
    torch.musa.synchronize()
    testing.DefaultComparator(res_amp, res_fp32)


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@pytest.mark.parametrize("low_dtype", [torch.bfloat16])
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_amp_autocast_bf16(low_dtype):
    res_fp32 = train_in_fp32()
    res_amp = train_in_amp(low_dtype=low_dtype)
    torch.musa.synchronize()
    testing.DefaultComparator(res_amp, res_fp32)
