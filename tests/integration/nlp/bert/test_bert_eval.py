"""Bert evaluation integration entry"""

import pytest

from transformers import AutoModelForSequenceClassification

from torch_musa.testing import get_musa_arch
from torch_musa.testing.integration import (
    IntegrationConfig,
    EvalConfig,
    JaccardScoreGreaterThan,
    default_strategies_list,
)

from .model import BertEvalIntegration
from .bert_utils import (
    get_bert_base_uncased_eval_small_imdb_model_dir,
    get_bert_base_uncased_eval_small_imdb_dataset,
)


def common_bert_base_uncased_run(
    batch_size,
    strategies,
    jaccard_score_lowerbound,
) -> None:
    """Build bert-base-uncased runner and run integration"""
    model_name = "bert-base-uncased"
    model_root = get_bert_base_uncased_eval_small_imdb_model_dir()
    eval_set = get_bert_base_uncased_eval_small_imdb_dataset(batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_root,
        local_files_only=False,
    )

    cfg = IntegrationConfig(
        model_name=model_name,
        strategies=strategies,
        conf=EvalConfig(
            batch_size=batch_size,
            eval_set=eval_set,
        ),
    )
    kwargs = {
        "assertions": {
            "eval_jaccard_score": JaccardScoreGreaterThan(jaccard_score_lowerbound),
        }
    }
    runner = BertEvalIntegration(cfg, model, **kwargs)
    runner.run()


@pytest.mark.skipif(get_musa_arch() < 21, reason="Only support arch greater equal 21")
@pytest.mark.parametrize("batch_size", [256])
@pytest.mark.parametrize("strategies", default_strategies_list())
def test_bert_base_uncased_common(batch_size, strategies) -> None:
    """Bert-base-uncased evaluation with arch >= 21"""
    common_bert_base_uncased_run(batch_size, strategies, jaccard_score_lowerbound=0.78)
