"""Quantitative metrics for evaluating the correctness of the integration process"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Any, Mapping
from time import monotonic

from .utils import Reference


@dataclass(frozen=True)
class NullOptT:
    pass


class MetricValueFormat(Enum):
    VALUE = 0
    DURATION_SECOND = auto()


class MetricValueFormatter:
    @staticmethod
    def default(value) -> str:
        return f"{value}"

    @staticmethod
    def translate_duration_second(value) -> str:
        return f"{round(value, 3)}s"


_MetricValueFormatterMap: Mapping[MetricValueFormat, Callable[[Any], str]] = {
    MetricValueFormat.VALUE: MetricValueFormatter.default,
    MetricValueFormat.DURATION_SECOND: MetricValueFormatter.translate_duration_second,
}


@dataclass
class Metric:
    """Preprocessed data collection categoried by same topic"""

    topic: str
    format: MetricValueFormat = MetricValueFormat.VALUE
    value: Any = NullOptT()

    def __post_init__(self):
        assert isinstance(self.topic, str) and (self.topic != "")

    def set_value(self, value: Any) -> None:
        self.value = value

    def str(self) -> str:
        if isinstance(self.value, NullOptT):
            message = "not assigned"
        else:
            message = _MetricValueFormatterMap[self.format](self.value)
        return f"{self.topic}: {message}"


class Duration:
    """Calculate process time within the global context"""

    _MEASURE: Callable[[], float] = monotonic
    _INVALID: float = -1.0

    def __init__(self, metric_ref: Reference) -> None:
        self.start = Duration._INVALID
        self.end = Duration._INVALID
        self.metric_ref = metric_ref

    def __enter__(self) -> "Duration":
        self.start = Duration._MEASURE()
        return self

    def __exit__(self, *exc_details):
        self.end = Duration._MEASURE()
        self.metric_ref.update(lambda x: x.set_value(self.end - self.start))
