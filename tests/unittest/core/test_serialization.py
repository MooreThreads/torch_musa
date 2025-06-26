"""Unit tests for serialization."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import os
import tempfile
import pytest

import torch

from torch import nn
import torch.nn.functional as F

import torch_musa


class ModelClass(nn.Module):
    """Sample model for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_module_save_load():
    """Test module save and load."""
    model = ModelClass()
    model = model.to("musa")
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.pt")
        torch.save(model, model_path)
        reloaded_model = torch.load(model_path, weights_only=False)
        # Ensure model loaded on MUSA
        assert (
            next(reloaded_model.parameters()).device == next(model.parameters()).device
        )

        # Ensure model loaded on CPU
        reloaded_model = torch.load(model_path, map_location="cpu", weights_only=False)
        assert next(reloaded_model.parameters()).device == torch.device("cpu")

        # Ensure raise readable error when model loaded to CUDA
        with pytest.raises(
            RuntimeError,
            match=r"Attempting to deserialize object on a CUDA device but "
            r"torch.cuda.is_available\(\) is False. If you are running on a MUSA enabled "
            r"machine, please use torch.load with map_location=torch.device\('musa'\) or "
            r"map_location=torch.device\('cpu'\) to map your storages to the MUSA or CPU.",
        ):
            reloaded_model = torch.load(
                model_path, map_location="cuda", weights_only=False
            )
