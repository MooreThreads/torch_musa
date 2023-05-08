"""Implement common function for test"""

import functools
from statistics import mean

import torch_musa


@functools.lru_cache()
def get_cycles_per_ms() -> float:
    """Measure and return approximate number of cycles per millisecond for torch_musa._sleep"""

    def measure() -> float:
        start = torch_musa.Event(enable_timing=True)
        end = torch_musa.Event(enable_timing=True)
        start.record()
        torch_musa._sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms

    # Get 10 values and remove the 2 max and 2 min and return the avg.
    # This is to avoid system disturbance that skew the results, e.g.
    # the very first musa call likely does a bunch of init, which takes
    # much longer than subsequent calls.

    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    return mean(vals[2 : num - 2])
