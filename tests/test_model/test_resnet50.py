"""Test resnet50."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, wrong-import-position, consider-using-from-import, not-callable
import sys

sys.path.insert(0, sys.path[0] + "/../unittest/operator")
from base_test_tool import DefaultComparator

import torch
import torchvision.models as models
from torchvision import datasets
from torch import Tensor
import pytest
import torch_musa

input_data = [
    {"input": torch.randn(1, 3, 224, 224)},
    {"input": torch.randn(1, 3, 256, 256)},
]


@pytest.mark.parametrize("input_data", input_data)
def test_rn50(input_data):
    rn50 = models.resnet50().to("musa")
    gpu_result = (
        rn50(torch.tensor(input_data["input"], device="musa", requires_grad=False))
        .cpu()
        .detach()
        .numpy()
    )
    cpu_result = (
        rn50.to("cpu")(torch.tensor(input_data["input"], requires_grad=False))
        .detach()
        .numpy()
    )
    comparator = DefaultComparator()
    assert comparator(gpu_result, cpu_result)
