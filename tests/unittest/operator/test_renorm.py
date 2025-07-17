"""Test renorm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data, p, dim, maxnorm",
    [
        (torch.randn(2, 2), 1, 0, 5),
        (torch.randn(3, 3, 3), 2, 1, 5),
        (torch.randn(4, 4, 4, 4), 2, 2, 5),
    ],
)
def test_renorm(input_data, p, dim, maxnorm):
    test = testing.OpTest(
        func=torch.renorm,
        input_args={"input": input_data, "p": p, "dim": dim, "maxnorm": maxnorm},
    )
    test.check_result()
    test.check_out_ops()
