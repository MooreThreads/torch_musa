"""Python utilities for integration"""

from .config import (
    TrainConfig,
    EvalConfig,
    IntegrationConfig,
)

from .metrics import (
    MetricValueFormat,
    Metric,
)

from .models import (
    TrainIntegration,
    EvalIntegration,
)

from .utils import (
    decompress,
    check_existent_directory,
    check_existent_file,
    get_dataset_root_dir,
    get_model_root_dir,
)

from .assertion import (
    Assertion,
    Top1AccuracyGreaterThan,
    Top5AccuracyGreaterThan,
    JaccardScoreGreaterThan,
    F1ScoreGreaterThan,
)

from .evalautor import *


def default_strategies_list():
    return [config.default_strategies()]
