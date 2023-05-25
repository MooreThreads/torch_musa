"""Test padnd operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from functools import partial
import pytest
import torch
from torch_musa import testing


all_support_types = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data",
    [
        torch.randn([1, 10], requires_grad=True),
        torch.randn([20, 30], requires_grad=True),
        torch.randn([1, 20, 40], requires_grad=True),
        torch.randn([1, 10, 30], requires_grad=True)
    ]
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("pad", [(3, 1), (2, 2)])
@pytest.mark.parametrize("mode", ["reflect"])
def test_pad1d(input_data, dtype, pad, mode):
    func = partial(torch.nn.functional.pad, pad=pad, mode=mode)
    cpu_input = input_data.to(dtype)
    musa_input = input_data.to(dtype)

    cpu_result = func(cpu_input)
    musa_result = func(musa_input.to("musa"))

    comparator = testing.DefaultComparator()
    assert comparator(cpu_result, musa_result.cpu())
    cpu_result.sum().backward()
    musa_result.sum().backward()
    assert comparator(cpu_input.grad, musa_input.grad.cpu())
