"""Implement common function for test"""

import contextlib
import functools
from statistics import mean

import pytest
import torch
from torch.testing._comparison import (
    BooleanPair,
    NonePair,
    not_close_error_metas,
    NumberPair,
    TensorLikePair,
)
from torch.utils._mode_utils import no_dispatch
import torch_musa
from torch_musa.core._utils import _get_musa_arch


def get_musa_arch() -> int:
    return _get_musa_arch()


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


def cpu_and_musa():
    return ("cpu", pytest.param("musa", marks=pytest.mark.needs_musa))


def needs_musa(test_func):
    return pytest.mark.needs_musa(test_func)


class ImagePair(TensorLikePair):
    """image pair definition"""

    def __init__(
        self,
        actual,
        expected,
        *,
        mae=False,
        **other_parameters,
    ):
        if all(isinstance(input, PIL.Image.Image) for input in [actual, expected]):
            actual, expected = [to_image_tensor(input) for input in [actual, expected]]

        super().__init__(actual, expected, **other_parameters)
        self.mae = mae

    def compare(self) -> None:
        actual, expected = self.actual, self.expected

        self._compare_attributes(actual, expected)
        actual, expected = self._equalize_attributes(actual, expected)

        if self.mae:
            if actual.dtype is torch.uint8:
                actual, expected = actual.to(torch.int), expected.to(torch.int)
            mae = float(torch.abs(actual - expected).float().mean())
            if mae > self.atol:
                self._fail(
                    AssertionError,
                    f"The MAE of the images is {mae}, but only {self.atol} is allowed.",
                )
        else:
            super()._compare_values(actual, expected)


def assert_close(
    actual,
    expected,
    *,
    allow_subclasses=True,
    rtol=None,
    atol=None,
    equal_nan=False,
    check_device=True,
    check_dtype=True,
    check_layout=True,
    check_stride=False,
    msg=None,
    **kwargs,
):
    """
    Superset of :func:`torch.testing.assert_close` with support for PIL vs.
    tensor image comparison
    """

    error_metas = not_close_error_metas(
        actual,
        expected,
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            ImagePair,
            TensorLikePair,
        ),
        allow_subclasses=allow_subclasses,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        **kwargs,
    )

    if error_metas:
        raise error_metas[0].to_error(msg)


assert_equal = functools.partial(assert_close, rtol=0, atol=0)
