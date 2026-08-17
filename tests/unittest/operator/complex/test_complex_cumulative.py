""" Tests for complex cumulative operations.

Covers the cumulative family of operators for complex64 / complex128:
    cumsum, cumprod, logcumsumexp,
    trapezoid, trapz, cumulative_trapezoid,
    masked.cumsum, masked.cumprod
"""

# pylint: disable=W0621, C0116

import math

import torch
import pytest
from torch_musa import testing


torch.manual_seed(41)

dtypes = [torch.complex64, torch.complex128]

# (shape, dim)
configs = [
    [(10,), 0],
    [(8, 16), 0],
    [(8, 16), 1],
    [(4, 8, 16), 2],
    [(4, 8, 16), -1],
    [(2, 3, 4, 5), 1],
]


def make_complex(shape, dtype, low=-2.0, high=2.0):
    """Build a complex tensor with non-trivial real and imaginary parts."""
    real = torch.empty(shape).uniform_(low, high)
    imag = torch.empty(shape).uniform_(low, high)
    return torch.complex(real, imag).to(dtype)


def comparator(dtype):
    if dtype == torch.complex64:
        return testing.DefaultComparator(abs_diff=1e-4, rel_diff=1e-4, equal_nan=True)
    return testing.DefaultComparator(abs_diff=1e-6, rel_diff=1e-6, equal_nan=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("func", [torch.cumsum, torch.cumprod])
def test_cumulative(config, dtype, func):
    """Tests for cumsum / cumprod."""
    shape, dim = config
    input_data = {
        "input": make_complex(shape, dtype),
        "dim": dim,
    }
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=comparator(dtype),
    )
    test.check_result()


def _assert_logcumsumexp_close(cpu, musa, dtype):
    """Compare complex logcumsumexp robustly against phase wrapping.

    The imaginary part is a phase (arg of the running sum). Near the negative
    real axis, CPU/GPU float noise makes atan2 jump between +pi and -pi -- a
    2*pi difference that is mathematically equivalent (exp() of both is equal).
    So compare the real part (log magnitude) directly, and the imaginary part
    modulo 2*pi (ring distance).
    """
    atol, rtol, _ = comparator(dtype).get_tolerance()
    assert cpu.shape == musa.shape
    assert torch.allclose(cpu.real, musa.real, atol=atol, rtol=rtol, equal_nan=True)
    two_pi = 2.0 * math.pi
    d = (cpu.imag - musa.imag).abs() % two_pi
    ring = torch.minimum(d, two_pi - d)
    tol = atol + rtol * musa.imag.abs() + 1e-4
    assert torch.all(ring <= tol), f"max phase diff (mod 2pi): {float(ring.max())}"


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("dtype", dtypes)
def test_logcumsumexp(config, dtype):
    """logcumsumexp with complex dtypes (phase-wrap-robust comparison)."""
    shape, dim = config
    inp = make_complex(shape, dtype)
    cpu_res = torch.logcumsumexp(inp, dim)
    musa_res = torch.logcumsumexp(inp.to("musa"), dim).cpu()
    _assert_logcumsumexp_close(cpu_res, musa_res, dtype)


@pytest.mark.skip(
    reason="blocked: muDNN where supports neither complex nor fp64 "
    "(real/imag split fails for complex128); needs ported where kernel"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("func", [torch.masked.cumsum, torch.masked.cumprod])
def test_masked_cumulative(config, dtype, func):
    """Tests for masked.cumsum / masked.cumprod."""
    shape, dim = config
    input_data = {
        "input": make_complex(shape, dtype),
        "dim": dim,
        "mask": torch.randint(0, 2, shape).bool(),
    }
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=comparator(dtype),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "func", [torch.trapezoid, torch.trapz, torch.cumulative_trapezoid]
)
def test_trapezoid(config, dtype, func):
    """Tests for trapezoid / trapz / cumulative_trapezoid."""
    shape, dim = config
    input_data = {
        "y": make_complex(shape, dtype),
        "dim": dim,
    }
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=comparator(dtype),
    )
    test.check_result()
