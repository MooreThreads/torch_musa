"""
Constant int benchmark res.
"""

from collections import namedtuple

# constant key
KEY_TEST_NAME = "test_name"
KEY_OP_NAME = "op"
KEY_TEST_CONFIG = "test_config"
KEY_LATENCY = "latency"
KEY_MEMORY = "memory"
KEY_GBS = "gb/s"
KEY_INTENSITY = "intensity"
KEY_TYPE = "type"
KEY_FLOPS = "flops"
KEY_TFLOPS = "tflops"
KEY_IS_BACKWARD = "backward"
KEY_OP_ATTR = "op_attr"
KEY_MODULE_TYPE = "op_type"
KEY_UNIT = "unit"
KEY_INPUT_CONFIG = "test_config"
KEY_TEST_CASES = "test_cases"
KEY_MODE = "mode"
KEY_TIME_METRIC = "time_metric"


# constant val
KEY_LATENCY_UNIT = "us"

TIME_METRIC = namedtuple("metric", "mean, var, percentile")
