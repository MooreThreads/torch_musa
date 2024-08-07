"""Resnet integration modules"""

from typing import Mapping, Optional

import torch

from torch_musa.testing.integration import (
    TrainIntegration,
    EvalIntegration,
    IntegrationConfig,
    Metric,
    Assertion,
    TopKAccuary,
)


class ResNetTrainIntegration(TrainIntegration):
    """Training integration implementation for resnet"""

    def __init__(
        self,
        conf: IntegrationConfig,
        model: torch.nn.Module,
        **args,
    ) -> None:
        super().__init__(conf, model)
        self.optimizer = args.pop("optimizer")
        self.lr_scheduler = args.pop("lr_scheduler", None)
        self.loss_fn = args.pop("loss_fn")

        assertions: Optional[Mapping[str, Assertion]] = args.pop("assertions", None)
        test_top1_accuracy_assertion = None
        test_top5_accuracy_assertion = None
        if assertions:
            assert isinstance(assertions, dict)
            test_top1_accuracy_assertion = assertions.get("test_top1_accuracy", None)
            test_top5_accuracy_assertion = assertions.get("test_top5_accuracy", None)

        self.register_metric(
            Metric(topic="test_top1_accuracy"), test_top1_accuracy_assertion
        )
        self.register_metric(
            Metric(topic="test_top5_accuracy"), test_top5_accuracy_assertion
        )
        self.evaluator = TopKAccuary([1, 5])

    def train_step(self, _, batch_datas) -> None:
        """Apply training forward and backward step"""
        device = self.device
        self.optimizer.zero_grad()
        inputs, labels = batch_datas[0].to(device), batch_datas[1].to(device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def test_step(self, _, batch_datas) -> None:
        """Apply evaluation forward step"""
        device = self.device
        inputs, labels = batch_datas[0].to(device), batch_datas[1].to(device)
        outputs = self.model(inputs)
        if not self.test_in_epoch:
            self.evaluator.append(torch.nn.functional.softmax(outputs, dim=1), labels)

    def prev_test(self) -> None:
        self.evaluator.zero_state()

    def post_test(self) -> None:
        top1_acc, top5_acc = self.evaluator.evaluate()
        self.update_metric_value("test_top1_accuracy", top1_acc)
        self.update_metric_value("test_top5_accuracy", top5_acc)


class ResNetEvalIntegration(EvalIntegration):
    """Evaluation integration implementation for resnet"""

    def __init__(
        self,
        conf: IntegrationConfig,
        model: torch.nn.Module,
        **args,
    ) -> None:
        super().__init__(conf, model)

        assertions: Optional[Mapping[str, Assertion]] = args.pop("assertions", None)
        eval_top1_accuracy_assertion = None
        eval_top5_accuracy_assertion = None
        if assertions:
            assert isinstance(assertions, dict)
            eval_top1_accuracy_assertion = assertions.get("eval_top1_accuracy", None)
            eval_top5_accuracy_assertion = assertions.get("eval_top5_accuracy", None)

        self.register_metric(
            Metric(topic="eval_top1_accuracy"), eval_top1_accuracy_assertion
        )
        self.register_metric(
            Metric(topic="eval_top5_accuracy"), eval_top5_accuracy_assertion
        )
        self.evaluator = TopKAccuary([1, 5])

    def eval_step(self, _, batch_datas) -> None:
        """Apply evaluation forward step"""
        device = self.device
        inputs, labels = batch_datas[0].to(device), batch_datas[1].to(device)
        outputs = self.model(inputs)
        self.evaluator.append(torch.nn.functional.softmax(outputs, dim=1), labels)

    def prev_eval(self) -> None:
        self.evaluator.zero_state()

    def post_eval(self) -> None:
        top1_acc, top5_acc = self.evaluator.evaluate()
        self.update_metric_value("eval_top1_accuracy", top1_acc)
        self.update_metric_value("eval_top5_accuracy", top5_acc)
