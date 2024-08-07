"""Test Module attributes"""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from copy import deepcopy
import torch
import pytest
from torchvision.models import (
    alexnet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)
import torch_musa
from torch_musa import testing


models = [
    x()
    for x in [
        alexnet,
        resnet18,
        resnet34,
        resnet50,
        resnet101,
        resnet152,
        vgg11,
        vgg11_bn,
        vgg13,
        vgg13_bn,
        vgg16,
        vgg16_bn,
        vgg19,
        vgg19_bn,
    ]
]


@pytest.mark.parametrize("model", models)
def test_module_musa(model):
    # `.musa`, `to("musa")` will modify model instance inplace hence we use copies
    model_cpy = deepcopy(model)
    model_musa = deepcopy(model).musa()
    model_to_musa = deepcopy(model).to("musa")

    for model_param, model_cpy_param, model_musa_param, model_to_musa_param in zip(
        model.parameters(),
        model_cpy.parameters(),
        model_musa.parameters(),
        model_to_musa.parameters(),
    ):
        assert torch.allclose(model_param, model_cpy_param)
        assert torch.allclose(model_param, model_musa_param.cpu())
        assert torch.allclose(model_musa_param, model_to_musa_param)

    if testing.MULTIGPU_AVAILABLE:
        model_musa_1 = deepcopy(model).musa(1)
        model_to_musa_1 = deepcopy(model).to("musa:1")
        for model_param, model_musa_1_param, model_to_musa_1_param in zip(
            model.parameters(), model_musa_1.parameters(), model_to_musa_1.parameters()
        ):
            assert torch.allclose(model_param, model_musa_1_param.cpu())
            assert torch.allclose(model_musa_1_param, model_to_musa_1_param)
