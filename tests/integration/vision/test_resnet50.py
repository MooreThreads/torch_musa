"""Test resnet50."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, wrong-import-position, consider-using-from-import, not-callable
import torch
import torchvision.models as models
import pytest
import torch_musa
from torch_musa import testing

INPUT_DATA = [
    {"input": torch.randn(1, 3, 224, 224)},
    {"input": torch.randn(1, 3, 256, 256)},
]


@pytest.mark.parametrize("input_data", INPUT_DATA)
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
    comparator = testing.DefaultComparator()
    assert comparator(gpu_result, cpu_result)
