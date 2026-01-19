"""Test arange operators."""

# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing
import torch_musa


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_assert():
    inps = [
        torch.tensor([1]).musa(),
        torch.tensor([1.0]).musa(),
        torch.tensor([1.0]).musa().to(torch.half),
        torch.tensor(1 + 2j).musa(),
        torch.tensor([True]).musa(),
    ]
    if testing.get_musa_arch() >= 22:
        inps.append(torch.tensor([1.0]).musa().to(torch.bfloat16))

    for inp in inps:
        torch._assert_async(inp)

    error_inps = [
        torch.tensor([0]).musa(),
        torch.tensor([0.0]).musa(),
        torch.tensor([0.0]).musa().to(torch.half),
        torch.tensor(0 + 0j).musa(),
        torch.tensor([False]).musa(),
    ]
    for error_inp in error_inps:
        with pytest.raises(
            RuntimeError,
            match="Expected Tensor with single nonzero value, but got zero",
        ):
            torch._assert_async(error_inp)
