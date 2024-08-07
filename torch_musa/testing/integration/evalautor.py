"""Evaluations between predicted and golden values"""

# pylint: disable = C0103
from abc import ABC, abstractmethod
from typing import Any, List

import torch

__all__ = [
    "JaccardSimilarityCoefficient",
    "F1Score",
    "TopKAccuary",
]


class Evaluator(ABC):
    """
    Evaluator receives multiple batches of data pairs (predictions and golden values)
    in a sequential order during training/testing/evaluation processes. After respective
    process completed, it aggregates all collected datas and calculates the metric value
    for later formatted output and assertion.
    Since some types of evaluations can perform per-batch calculation during the runtime
    process, others may need to delay the calculation until the entire process finished.
    class evaluator tries to encapsulate this difference and provide unified interfaces
    for various metric calculations.
    """

    @abstractmethod
    def append(self, prediction: torch.Tensor, golden: torch.Tensor) -> None: ...

    @abstractmethod
    def evaluate(self) -> Any: ...

    @abstractmethod
    def zero_state(self) -> None: ...


class JaccardSimilarityCoefficient(Evaluator):
    """
    Binary classification accuracy with two length-N attributes.
    Calculation formula: TP / (TP + FP + FN)
    """

    def __init__(self):
        self.tp = 0
        self.fp_tp = 0
        self.fn_tp = 0

    def append(self, prediction: torch.Tensor, golden: torch.Tensor) -> None:
        """
        Prediction should already be processed by argmax function.
        """
        mask = golden.gt(0)
        self.tp += (prediction.eq(golden) * mask).sum().item()
        self.fp_tp += prediction.gt(0).sum().item()
        self.fn_tp += golden.gt(0).sum().item()

    def evaluate(self):
        score = self.tp / (self.fn_tp + self.fp_tp - self.tp)
        return score

    def zero_state(self) -> None:
        self.tp = 0
        self.fp_tp = 0
        self.fn_tp = 0


class F1Score(Evaluator):
    """
    Traditional balanced F-score (the harmonic mean of precision and recall)
    Calculation formula: (2 * TP) / (2 * TP + FP + FN)
    """

    def __init__(self):
        self.tp = 0
        self.fp_tp = 0
        self.fn_tp = 0

    def append(self, prediction: torch.Tensor, golden: torch.Tensor) -> None:
        """
        Prediction should already be processed by argmax function.
        """
        mask = golden.gt(0)
        self.tp += (prediction.eq(golden) * mask).sum().item()
        self.fp_tp += prediction.gt(0).sum().item()
        self.fn_tp += golden.gt(0).sum().item()

    def evaluate(self):
        score = self.tp / (self.fn_tp + self.fp_tp)
        return score * 2

    def zero_state(self) -> None:
        self.tp = 0
        self.fp_tp = 0
        self.fn_tp = 0


class TopKAccuary(Evaluator):
    """
    In multi-class classification tasks, it takes into account whether the
    correct label is present in the top-k preditions.
    """

    def __init__(self, ks: List[int]):
        self.ks = ks
        self.total = 0
        self.correct = [0] * len(self.ks)

    def append(self, prediction: torch.Tensor, golden: torch.Tensor) -> None:
        """
        Prediction should already be processed by softmax function.
        """
        self.total += golden.size(0)
        for i, k in enumerate(self.ks):
            _, topk_idxes = prediction.topk(k, dim=1)
            self.correct[i] += (golden.view(-1, 1) == topk_idxes).sum().item()

    def evaluate(self):
        return [c / self.total for c in self.correct]

    def zero_state(self) -> None:
        self.total = 0
        self.correct = [0] * len(self.ks)
