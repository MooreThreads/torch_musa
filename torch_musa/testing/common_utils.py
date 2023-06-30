"""Implement common function for test"""

import functools
import contextlib
from statistics import mean
from torch.utils._mode_utils import no_dispatch
import torch

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


@contextlib.contextmanager
def disable_functorch():
    guard = torch._C._DisableFuncTorch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


@contextlib.contextmanager
def freeze_rng_state():
    """
    no_dispatch needed for test_composite_compliance
    Some OpInfos use freeze_rng_state for rng determinism, but
    test_composite_compliance overrides dispatch for all torch functions
    which we need to disable to get and set rng state
    """
    with no_dispatch(), disable_functorch():
        rng_state = torch.get_rng_state()
        if torch_musa.is_available():
            musa_rng_state = torch_musa.get_rng_state()
    try:
        yield
    finally:
        # Modes are not happy with torch.musa.set_rng_state
        # because it clones the state (which could produce a Tensor Subclass)
        # and then grabs the new tensor's data pointer in generator.set_state.
        #
        # In the long run torch.musa.set_rng_state should probably be
        # an operator.
        #
        # NB: Mode disable is to avoid running cross-ref tests on thes seeding
        with no_dispatch(), disable_functorch():
            if torch_musa.is_available():
                torch_musa.set_rng_state(musa_rng_state)
            torch.set_rng_state(rng_state)
