r"""
Supported modified profiler utilities
"""

import torch
from ._pattern_matcher import (
    ExtraMUSACopyPattern,
    FP32MatMulPattern,
    MatMulDimInFP16Pattern,
    report_all_anti_patterns,
)
from ._utils import BasicEvaluation


def set_profiler_attributes():
    """Set torch profiler attributes for torch musa."""
    setattr(torch.profiler._utils, "BasicEvaluation", BasicEvaluation)
    setattr(
        torch.profiler._pattern_matcher, "ExtraMUSACopyPattern", ExtraMUSACopyPattern
    )
    setattr(torch.profiler._pattern_matcher, "FP32MatMulPattern", FP32MatMulPattern)
    setattr(
        torch.profiler._pattern_matcher,
        "MatMulDimInFP16Pattern",
        MatMulDimInFP16Pattern,
    )
    setattr(
        torch.profiler._pattern_matcher,
        "report_all_anti_patterns",
        report_all_anti_patterns,
    )
