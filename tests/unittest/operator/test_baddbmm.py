"""Test baddbmm operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

input_data = [
    {
        "input": torch.randn(4, 5, 2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": 1,
        "alpha": 1
    },
    {
        "input": torch.randn(4, 5, 2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": 2.4,
        "alpha": 3.2
    },
    {
        "input": torch.randn(2),
        "batch1": torch.randn(4, 5, 10),
        "batch2": torch.randn(4, 10, 2),
        "beta": -2.4,
        "alpha": 3.2
    },
]

@pytest.mark.parametrize("input_data", input_data)
def test_baddbmm(input_data):
    test = testing.OpTest(
        func=torch.baddbmm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-5)
    )
    test.check_result()

@testing.skip_if_not_multiple_musa_device
def test_baddbmm_device():
    cpu_input = torch.randn(15, 2)
    cpu_batch1 = torch.randn(4, 15, 12)
    cpu_batch2 = torch.randn(4, 12, 2)
    beta = 1.6
    alpha = 1.3
    cpu_result = torch.baddbmm(input=cpu_input,
                               batch1=cpu_batch1,
                               batch2=cpu_batch2,
                               beta=beta,
                               alpha=alpha)

    musa_input = cpu_input.to("musa:1")
    musa_batch1 = cpu_batch1.to("musa:1")
    musa_batch2 = cpu_batch2.to("musa:1")
    musa_result = torch.baddbmm(input=musa_input,
                                batch1=musa_batch1,
                                batch2=musa_batch2,
                                beta=beta,
                                alpha=alpha)

    assert testing.DefaultComparator(1e-5)(musa_result.cpu(), cpu_result)
    assert musa_result.shape == cpu_result.shape
    assert musa_result.dtype == cpu_result.dtype
