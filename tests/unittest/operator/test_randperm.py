"""Test random operators."""
# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing

import torch_musa

generator = [torch.Generator()]
n = [5, 100, 50000, 100000]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("n", n)
@pytest.mark.parametrize("device", ['musa'])
def test_randperm_n_device(n, device):
    assert n == torch.randperm(n, device=device).numel()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("n", n)
@pytest.mark.parametrize("device", ['cpu', 'musa'])
@pytest.mark.parametrize("dtype", testing.get_all_support_types())
def test_randperm_n_device_dtype(n, device, dtype):
    torch.randperm(n, device=device, dtype=dtype)


n = [4, 5, 6, 10, 20]
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("n", n)
@pytest.mark.parametrize("device", ['musa'])
def test_randperm_n_device_value(n, device):
    assert torch.all(torch.eq(torch.randperm(n, device=device).sort().values.long(),
                              torch.arange(n, device=device).long()))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("n", n)
def test_randperm_n(n):
    input_params = {
            'n': n}
    testing.OpTest(
        func=torch.randperm,
        input_args=input_params,
    )

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("n", n)
@pytest.mark.parametrize("dtype", testing.get_all_support_types())
def test_randperm_n_musa_dtype(n, dtype):
    input_params = {
            'n': n,
            'dtype': dtype,
            'device': 'musa'}
    testing.OpTest(
        func=torch.randperm,
        input_args=input_params,
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("n", n)
@pytest.mark.parametrize("generator", generator)
def test_randperm_generator(n, generator):
    input_params = {
            'n': n,
            'generator': generator}
    testing.OpTest(
        func=torch.randperm,
        input_args=input_params,
    )
