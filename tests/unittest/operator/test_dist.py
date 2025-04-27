"""Test cdist forward & backward operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
from torch.nn import functional as F

import torch_musa
from torch_musa import testing


def cdist_input_shapes():
    return [
        [[100, 10], [100, 10]],  # small
        [[1024, 10], [128, 10]],  # large
        [[4, 3, 1024, 10], [4, 3, 128, 10]],  # batch
    ]


def pdist_input_shapes():
    return [[100, 10], [1024, 128], [2048, 512]]  # small  # medium  # large


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_ts", cdist_input_shapes())
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("p", [1.0, 2.0, torch.inf])
@pytest.mark.parametrize(
    "compute_mode",
    [
        "use_mm_for_euclid_dist_if_necessary",
        "use_mm_for_euclid_dist",
        "donot_use_mm_for_euclid_dist",
    ],
)
def test_cdist(input_ts, dtype, p, compute_mode):

    x_1 = torch.randn(input_ts[0], dtype=dtype, requires_grad=True)
    x_2 = torch.randn(input_ts[1], dtype=dtype, requires_grad=True)

    input_args = {
        "x1": x_1.musa(),
        "x2": x_2.musa(),
        "p": p,
        "compute_mode": compute_mode,
    }

    atol, rtol = 1e-6, 1e-6
    if p == 2.0:
        atol = 1e-3

    test = testing.OpTest(
        func=torch.cdist,
        input_args=input_args,
        comparators=testing.DefaultComparator(abs_diff=atol, rel_diff=rtol),
    )
    test.check_result(train=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", pdist_input_shapes())
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("p", [1.0, 2.0])
def test_pdist(input_shape, dtype, p):
    input_ts = torch.randn(input_shape, dtype=dtype, requires_grad=True)
    input_args = {
        "input": input_ts.musa(),
        "p": p,
    }

    atol, rtol = 1e-6, 1e-6
    if p == 2.0:
        atol = 1e-3

    test = testing.OpTest(
        func=F.pdist,
        input_args=input_args,
        comparators=testing.DefaultComparator(abs_diff=atol, rel_diff=rtol),
    )
    test.check_result(train=True)
