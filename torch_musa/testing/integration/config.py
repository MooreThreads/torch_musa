"""Basic training/evaluation integration configs"""

from dataclasses import dataclass, field
from typing import TypeVar, Any, List, Union

from .utils import assert_never

_T = TypeVar("_T")


def _required() -> _T:
    f: _T

    def _throw():
        raise ValueError(f"Field `{f.name}` required")

    f = field(default_factory=_throw)
    return f


def default_strategies() -> List:
    return []


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 2
    max_epochs: int = 2
    train_set: Any = _required()
    test_set: Any = _required()

    def summary(self) -> str:
        return f"batch_size:{self.batch_size}-max_epochs:{self.max_epochs}"


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int = 1
    eval_set: Any = _required()

    def summary(self) -> str:
        return f"batch_size:{self.batch_size}"


@dataclass(frozen=True)
class IntegrationConfig:
    """Top-level configuration for integration process"""

    model_name: str = _required()
    model_file: str = ""
    device: str = "musa"
    strategies: List[str] = field(default_factory=default_strategies)
    seed: int = 2034
    conf: Union[TrainConfig, EvalConfig] = _required()

    def infer_mode(self) -> str:
        if isinstance(self.conf, TrainConfig):
            return "train"
        if isinstance(self.conf, EvalConfig):
            return "eval"
        assert_never(self.conf)

    def summary(self) -> str:
        mode = self.infer_mode()
        sub_summary = self.conf.summary()
        return f"{self.model_name}-{mode}-{self.device}-strategies:{self.strategies}-{sub_summary}"
