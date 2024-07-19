"""Stateful metric checkers"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar


class Assertion(ABC):
    @abstractmethod
    def check(self, result: Any) -> None: ...


# pylint: disable=C0103
ComparableType = TypeVar("ComparableType", int, float)


class ComparableGreaterThan(Assertion):
    def __init__(self, lowerbound: ComparableType) -> None:
        self.lowerbound = lowerbound

    def check(self, result: ComparableType) -> None:
        assert result > self.lowerbound


Top1AccuracyGreaterThan = type("Top1AccuracyGreaterThan", (ComparableGreaterThan,), {})
Top5AccuracyGreaterThan = type("Top5AccuracyGreaterThan", (ComparableGreaterThan,), {})
JaccardScoreGreaterThan = type("JaccardScoreGreaterThan", (ComparableGreaterThan,), {})
F1ScoreGreaterThan = type("F1ScoreGreaterThan", (ComparableGreaterThan,), {})
