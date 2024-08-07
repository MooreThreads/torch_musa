"""Resnet evaluation integration entry"""

import pytest

from torchvision.models import resnet

from torch_musa.testing import get_musa_arch
from torch_musa.testing.integration import (
    IntegrationConfig,
    EvalConfig,
    Top1AccuracyGreaterThan,
    Top5AccuracyGreaterThan,
    default_strategies_list,
)

from .resnet_utils import (
    get_imagenet2012_evaluation_small_dataset,
    get_resnet50_evaluation_ckpt,
)
from .model import ResNetEvalIntegration


def common_resnet50_run(
    batch_size,
    strategies,
    top1_acc_lowerbound,
    top5_acc_lowerbound,
) -> None:
    """Build reset50 runner and run integration"""
    model_name = "resnet50"
    model = getattr(resnet, model_name)()
    eval_set = get_imagenet2012_evaluation_small_dataset(batch_size)
    cfg = IntegrationConfig(
        model_name=model_name,
        model_file=get_resnet50_evaluation_ckpt(),
        strategies=strategies,
        conf=EvalConfig(
            batch_size=batch_size,
            eval_set=eval_set,
        ),
    )
    kwargs = {
        "assertions": {
            "eval_top1_accuracy": Top1AccuracyGreaterThan(top1_acc_lowerbound),
            "eval_top5_accuracy": Top5AccuracyGreaterThan(top5_acc_lowerbound),
        }
    }
    runner = ResNetEvalIntegration(cfg, model, **kwargs)
    runner.run()


def resnet50_common_strategies():
    strategies = default_strategies_list()
    strategies.append(["channelslast_2d"])
    return strategies


@pytest.mark.skipif(get_musa_arch() < 21, reason="Only support arch greater equal 21")
@pytest.mark.parametrize("batch_size", [256])
@pytest.mark.parametrize("strategies", resnet50_common_strategies())
def test_resnet50_common(batch_size, strategies) -> None:
    """Resnet50 evaluation with arch >= 21"""
    common_resnet50_run(
        batch_size, strategies, top1_acc_lowerbound=0.76, top5_acc_lowerbound=0.93
    )
