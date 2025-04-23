"""Utility functions for inductor"""

# pylint: disable=import-outside-toplevel
import functools
import inspect
import time

import torch
from torch._inductor.utils import synchronize


@functools.lru_cache(None)
def triton_is_available():
    if not torch.musa.is_available():
        return False
    try:
        import triton

        return triton is not None
    except ImportError:
        return False


def timed(model, example_inputs, times=1):
    """time the model's ntimes"""
    synchronize("musa")
    torch.manual_seed(1337)
    torch.musa.manual_seed(1337)
    start_time = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize("musa")
    end_time = time.perf_counter()
    # GC the result after timing
    assert result is not None
    return end_time - start_time


def print_performance(func, args=(), times=10, repeat=10, baseline=1.0):
    timings = torch.tensor([timed(func, args, times) for _ in range(repeat)])
    took = torch.median(timings) / times
    print(f"{took/baseline:.6f}")
    return took


def do_bench(*args, **kwargs):
    """benchmark the generated triton kernel"""

    @functools.lru_cache(None)
    def load_triton():
        try:
            # import from triton.musa_testing in triton_musa-2.1.0
            from triton.musa_testing import do_bench as triton_do_bench

            # from triton.backends.mtgpu.musa_testing import do_bench as triton_do_bench
        except ImportError as exc:
            raise NotImplementedError("requires Triton") from exc
        # triton PR https://github.com/openai/triton/pull/1513 change the
        # quantile fields name from 'percentiles' to 'quantiles'
        # and change the default value from (0.5, 0.2, 0.8) to None.
        # This may break inductor since a caller expects a tuple may get a item.
        #
        # Add a wrapper to maintain the same behavior for inductor.
        # Maybe we should have own implementation of this function?
        return triton_do_bench, (
            "quantiles"
            if inspect.signature(triton_do_bench).parameters.get("quantiles")
            is not None
            else "percentiles"
        )

    triton_do_bench, quantile_field_name = load_triton()

    if quantile_field_name not in kwargs:
        kwargs[quantile_field_name] = (0.5, 0.2, 0.8)
    return triton_do_bench(*args, **kwargs)[0]
