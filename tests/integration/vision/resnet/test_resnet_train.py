"""Resnet training integration entry"""

import pytest

import torch
from torchvision.models import resnet

from torch_musa.testing import get_musa_arch
from torch_musa.testing.integration import (
    IntegrationConfig,
    TrainConfig,
    Top1AccuracyGreaterThan,
    Top5AccuracyGreaterThan,
    default_strategies_list,
)

from .resnet_utils import get_resnet50_training_ckpt, get_cifar10_training_dataset
from .model import ResNetTrainIntegration


def common_resnet50_run(
    batch_size,
    max_epochs,
    strategies,
    top1_acc_lowerbound,
    top5_acc_lowerbound,
) -> None:
    """Build reset50 runner and run integration"""
    model_name = "resnet50"
    model = getattr(resnet, model_name)()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_set, test_set = get_cifar10_training_dataset(batch_size)
    cfg = IntegrationConfig(
        model_name=model_name,
        model_file=get_resnet50_training_ckpt(),
        strategies=strategies,
        conf=TrainConfig(
            batch_size=batch_size,
            max_epochs=max_epochs,
            train_set=train_set,
            test_set=test_set,
        ),
    )
    kwargs = {
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "assertions": {
            "test_top1_accuracy": Top1AccuracyGreaterThan(top1_acc_lowerbound),
            "test_top5_accuracy": Top5AccuracyGreaterThan(top5_acc_lowerbound),
        },
    }
    runner = ResNetTrainIntegration(cfg, model, **kwargs)
    runner.run()


def resnet50_common_strategies():
    strategies = default_strategies_list()
    strategies.append(["channelslast_2d"])
    return strategies


@pytest.mark.skipif(get_musa_arch() < 21, reason="Only support arch greater equal 21")
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("max_epochs", [2])
@pytest.mark.parametrize("strategies", resnet50_common_strategies())
def test_resnet50_common(batch_size, max_epochs, strategies) -> None:
    """Resnet50 training with arch >= 21"""
    common_resnet50_run(
        batch_size,
        max_epochs,
        strategies,
        top1_acc_lowerbound=0.82,
        top5_acc_lowerbound=0.98,
    )
