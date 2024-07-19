"""Basic integration model"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
from enum import auto
import logging
from typing import Mapping, Union, Any, Tuple, Optional

import torch

from .assertion import Assertion
from .config import IntegrationConfig
from . import strategy
from .utils import ExtendedEnum, assert_never, check_device, set_seed, Reference
from .metrics import Metric, Duration, MetricValueFormat


class Strategy(ExtendedEnum):
    TF32 = 0
    AMP = auto()
    CHANNELSLAST_2D = auto()

    def __str__(self) -> str:
        return self.name.lower()


_SupportStrategies = Strategy.choices()
_TextStrategyMap = Strategy.text_map()


def text_to_strategy(text: str) -> Strategy:
    clean_text: str = text.strip().lower()
    if clean_text not in _TextStrategyMap:
        raise ValueError(
            f"Invalid strategy `{text}`, expect one of {_SupportStrategies}"
        )
    return _TextStrategyMap[clean_text]


class Integration(ABC):
    """Common integration framework"""

    def __init__(
        self,
        conf: IntegrationConfig,
        model: torch.nn.Module,
    ) -> None:
        self.conf = conf
        self.model = model

        check_device(self.conf.device)
        self.device = torch.device(self.conf.device)

        self.global_strategies: list[strategy.GlobalStrategy] = []
        self.model_strategies: list[strategy.ModelStrategy] = []
        self.runtime_strategies: list[strategy.RuntimeStrategy] = []
        for text in self.conf.strategies:
            mode = text_to_strategy(text)
            if mode == Strategy.TF32:
                self.global_strategies.append(strategy.TF32())
            elif mode == Strategy.AMP:
                self.runtime_strategies.append(strategy.AMP())
            elif mode == Strategy.CHANNELSLAST_2D:
                self.model_strategies.append(strategy.ChannelsLast2D())
            else:
                assert_never(mode)

        self.logger = logging.getLogger(f"{self.conf.summary()} logger")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
            )
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(handler)

        set_seed(self.conf.seed)

        self.metrics_group: Mapping[str, Tuple[Reference, Optional[Assertion]]] = (
            OrderedDict()
        )
        self.register_metric(
            Metric(topic="running_time", format=MetricValueFormat.DURATION_SECOND)
        )

    @abstractmethod
    def mode(self) -> str: ...

    def register_metric(
        self,
        metric: Union[Metric, Reference],
        assertion: Optional[Assertion] = None,
    ) -> None:
        """Record metric with optional assertion for the same topic"""
        topic: str = None
        if isinstance(metric, Metric):
            topic = metric.topic
            self.metrics_group[topic] = (Reference(metric), assertion)
        elif isinstance(metric, Reference):
            topic = metric.current().topic
            self.metrics_group[topic] = (metric, assertion)
        else:
            assert_never(metric)

    def get_metric(self, topic: str) -> Reference:
        if topic not in self.metrics_group:
            raise ValueError(f"`{topic}` metric used before registered")
        return self.metrics_group[topic][0]

    def get_metric_value(self, topic: str) -> Any:
        return self.get_metric(topic).current().value

    def update_metric_value(self, topic: str, value: Any) -> None:
        self.get_metric(topic).update(lambda x: x.set_value(value))

    def register_assertion(self, topic: str, assertion: Assertion) -> None:
        if topic not in self.metrics_group:
            raise ValueError(
                f"`{topic}` assertion registered before corresponding metric"
            )
        self.metrics_group[topic][1] = assertion

    def config_summary(self) -> None:
        self.logger.info("Task: %s", self.conf.summary())

    def load_model(self) -> None:
        if self.conf.model_file != "":
            self.model.load_state_dict(
                torch.load(self.conf.model_file, map_location=self.device)
            )

    def transform_model(self) -> None:
        self.model.to(self.device)
        for model_strategy in self.model_strategies:
            self.model = model_strategy.transform(self.model)

    @abstractmethod
    def inner_run(self) -> None: ...

    def run(self) -> None:
        """Top integration processing entry"""
        with ExitStack() as stack:
            stack.enter_context(Duration(self.get_metric("running_time")))
            self.load_model()
            self.transform_model()
            for global_strategy in self.global_strategies:
                stack.enter_context(global_strategy.make_context())
            self.inner_run()
        self.config_summary()
        self.metrics_summary()
        self.do_assert()

    def metrics_summary(self) -> None:
        for reference, _ in self.metrics_group.values():
            metric = reference.current()
            self.logger.info(metric.str())

    def do_assert(self) -> None:
        for reference, assertion in self.metrics_group.values():
            if assertion:
                assertion.check(reference.current().value)


class TrainIntegration(Integration):
    """Basic training integration framework"""

    def __init__(
        self,
        conf: IntegrationConfig,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(conf, model)
        self.train_conf = self.conf.conf
        self.test_in_epoch = False
        assert self.conf.infer_mode() == self.mode()

    def mode(self) -> str:
        return "train"

    def prev_epoch_train(self) -> None:
        pass

    def post_epoch_train(self) -> None:
        pass

    def prev_epoch_test(self) -> None:
        pass

    def post_epoch_test(self) -> None:
        pass

    def prev_test(self) -> None:
        pass

    def post_test(self) -> None:
        pass

    @abstractmethod
    def train_step(self, batch_idx, batch_datas) -> None: ...

    @abstractmethod
    def test_step(self, batch_idx, batch_datas) -> None: ...

    def inner_run(self) -> None:
        """Basic training process"""
        train_set = self.train_conf.train_set
        test_set = self.train_conf.test_set

        def mark_train() -> None:
            self.model.train()
            torch.set_grad_enabled(True)

        def mark_eval() -> None:
            self.model.eval()
            torch.set_grad_enabled(False)

        self.logger.info("Running training...")
        self.test_in_epoch = True
        with ExitStack() as stack:
            for runtime_strategy in self.runtime_strategies:
                stack.enter_context(runtime_strategy.make_context())
            for epoch in range(self.train_conf.max_epochs):
                mark_train()
                self.logger.info("Training epoch %d...", epoch)
                self.prev_epoch_train()
                for batch_idx, batch_datas in enumerate(train_set):
                    self.train_step(batch_idx, batch_datas)
                self.post_epoch_train()
                mark_eval()
                self.prev_epoch_test()
                self.logger.info("Testing epoch %d...", epoch)
                for batch_idx, batch_datas in enumerate(test_set):
                    self.test_step(batch_idx, batch_datas)
                self.post_epoch_test()
        self.logger.info("Running validation...")
        self.test_in_epoch = False
        with torch.inference_mode(mode=True):
            self.prev_test()
            for batch_idx, batch_datas in enumerate(test_set):
                self.test_step(batch_idx, batch_datas)
            self.post_test()


class EvalIntegration(Integration):
    """Basic evaluation integration framework"""

    def __init__(
        self,
        conf: IntegrationConfig,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(conf, model)
        self.eval_conf = self.conf.conf
        assert self.conf.infer_mode() == self.mode()

    def mode(self) -> str:
        return "eval"

    def prev_eval(self) -> None:
        pass

    def post_eval(self) -> None:
        pass

    @abstractmethod
    def eval_step(self, batch_idx, batch_datas) -> None: ...

    def inner_run(self) -> None:
        """Basic evaluation process"""
        eval_set = self.eval_conf.eval_set

        self.model.eval()
        torch.set_grad_enabled(False)
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode(mode=True))
            for runtime_strategy in self.runtime_strategies:
                stack.enter_context(runtime_strategy.make_context())
            self.logger.info("Running evaluation...")
            self.prev_eval()
            for batch_idx, batch_datas in enumerate(eval_set):
                self.eval_step(batch_idx, batch_datas)
            self.post_eval()
