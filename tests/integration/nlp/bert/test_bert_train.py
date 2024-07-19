"""Bert training integration entry"""

import pytest

import torch
from transformers import (
    get_scheduler,
    AutoModelForSequenceClassification,
)

from torch_musa.testing import get_musa_arch
from torch_musa.testing.integration import (
    IntegrationConfig,
    TrainConfig,
    JaccardScoreGreaterThan,
    default_strategies_list,
)

from .model import BertTrainIntegration
from .bert_utils import (
    get_bert_base_uncased_train_small_imdb_model_dir,
    get_bert_base_uncased_train_small_imdb_datasets,
)


def common_bert_base_uncased_run(
    batch_size,
    max_epochs,
    strategies,
    jaccard_score_lowerbound,
) -> None:
    """Build bert-base-uncased runner and run integration"""
    model_name = "bert-base-uncased"
    model_root = get_bert_base_uncased_train_small_imdb_model_dir()
    train_set, test_set = get_bert_base_uncased_train_small_imdb_datasets(batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_root,
        local_files_only=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_epochs * len(train_set),
    )

    cfg = IntegrationConfig(
        model_name=model_name,
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
        "lr_scheduler": lr_scheduler,
        "assertions": {
            "test_jaccard_score": JaccardScoreGreaterThan(jaccard_score_lowerbound),
        },
    }
    runner = BertTrainIntegration(cfg, model, **kwargs)
    runner.run()


@pytest.mark.skipif(get_musa_arch() < 21, reason="Only support arch greater equal 21")
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("max_epochs", [1])
@pytest.mark.parametrize("strategies", default_strategies_list())
def test_bert_base_uncased_common(batch_size, max_epochs, strategies) -> None:
    """Bert-base-uncased training with arch >= 21"""
    device_type_name = torch.musa.get_device_name(torch.musa.current_device())
    if device_type_name == "MTT S80":  # skip this test in S80
        pytest.skip(
            reason="Musa Mem may be out-of-memory in S80 for this bert train test."
        )
    common_bert_base_uncased_run(
        batch_size, max_epochs, strategies, jaccard_score_lowerbound=0.77
    )
