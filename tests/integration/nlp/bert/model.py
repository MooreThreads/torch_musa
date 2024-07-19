"""Bert integration modules"""

from typing import Mapping, Optional

import torch

from torch_musa.testing.integration import (
    TrainIntegration,
    EvalIntegration,
    IntegrationConfig,
    Metric,
    Assertion,
    JaccardSimilarityCoefficient,
)


class BertTrainIntegration(TrainIntegration):
    """Training integration implementation for bert"""

    def __init__(
        self,
        conf: IntegrationConfig,
        model: torch.nn.Module,
        **args,
    ) -> None:
        super().__init__(conf, model)
        self.optimizer = args.pop("optimizer")
        self.lr_scheduler = args.pop("lr_scheduler")

        assertions: Optional[Mapping[str, Assertion]] = args.pop("assertions", None)
        test_jaccard_score_assertion = None
        if assertions:
            assert isinstance(assertions, dict)
            test_jaccard_score_assertion = assertions.get("test_jaccard_score", None)

        self.register_metric(
            Metric(topic="test_jaccard_score"), test_jaccard_score_assertion
        )
        self.evaluator = JaccardSimilarityCoefficient()

    def train_step(self, _, batch_datas) -> None:
        """Apply training forward and backward step"""
        self.optimizer.zero_grad()
        inputs = {k: v.to(self.device) for k, v in batch_datas.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def test_step(self, _, batch_datas) -> None:
        """Apply evaluation forward step"""
        inputs = {k: v.to(self.device) for k, v in batch_datas.items()}
        outputs = self.model(**inputs)
        if not self.test_in_epoch:
            self.evaluator.append(
                prediction=torch.argmax(outputs.logits, dim=-1),
                golden=batch_datas["labels"].to(self.device),
            )

    def prev_test(self) -> None:
        self.evaluator.zero_state()

    def post_test(self) -> None:
        self.update_metric_value("test_jaccard_score", self.evaluator.evaluate())


class BertEvalIntegration(EvalIntegration):
    """Evaluation integration implementation for bert"""

    def __init__(
        self,
        conf: IntegrationConfig,
        model: torch.nn.Module,
        **args,
    ) -> None:
        super().__init__(conf, model)

        assertions: Optional[Mapping[str, Assertion]] = args.pop("assertions", None)
        eval_jaccard_score_assertion = None
        if assertions:
            assert isinstance(assertions, dict)
            eval_jaccard_score_assertion = assertions.get("eval_jaccard_score", None)

        self.register_metric(
            Metric(topic="eval_jaccard_score"), eval_jaccard_score_assertion
        )
        self.evaluator = JaccardSimilarityCoefficient()

    def eval_step(self, _, batch_datas) -> None:
        """Apply evaluation forward step"""
        inputs = {k: v.to(self.device) for k, v in batch_datas.items()}
        outputs = self.model(**inputs)
        self.evaluator.append(
            prediction=torch.argmax(outputs.logits, dim=-1),
            golden=batch_datas["labels"].to(self.device),
        )

    def prev_eval(self) -> None:
        self.evaluator.zero_state()

    def post_eval(self) -> None:
        score = self.evaluator.evaluate()
        self.update_metric_value("eval_jaccard_score", score)
