# pylint: disable= missing-module-docstring, missing-class-docstring,missing-function-docstring,unused-import,unused-variable,not-callable,C0103
import os
import random
import pytest
import torch
from torch import nn
import numpy as np
import torch_musa
from torch_musa import testing

DEVICE = "musa"

def set_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)
    torch.musa.manual_seed_all(seed)


class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=True)
        self.fc = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 32, 10)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_simple_conv_model(monkeypatch):
    monkeypatch.setenv("TORCH_USE_MUSA_DOUBLE_CAST", "1")
    simple_conv_model = SimpleConvModel().to(DEVICE).double()
    x_musa = torch.randn(2, 3, 32, 32, dtype=torch.double, device=DEVICE)
    with torch.no_grad():
        output = simple_conv_model(x_musa)
    assert output.dtype == torch.float64, "cast fail"


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_simple_transformer_model(monkeypatch):
    monkeypatch.setenv("TORCH_USE_MUSA_DOUBLE_CAST", "1")
    simple_transformer_model = SimpleTransformer().to(DEVICE).double()
    x_musa = torch.randn(2, 16, 32, dtype=torch.double, device=DEVICE)
    with torch.no_grad():
        output = simple_transformer_model(x_musa)
    assert output.dtype == torch.float64, "cast fail"
