"""Test random operators."""
# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing

import torch_musa

size = [(2,), (2, 3), (2, 3, 4)]
low = [1, 10]
high = [20, 30]

generator = [torch.Generator()]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("low", low)
@pytest.mark.parametrize("high", high)
@pytest.mark.parametrize("generator", generator)
def test_randint_low_generator(low, high, size, generator):
    input_params = {
            'low': low,
            'high': high,
            'size': size,
            'generator': generator,
            'device': 'musa'}
    test = testing.OpTest(
        func=torch.randint,
        input_args=input_params,
        ignored_result_indices=[0]
    )
    test.check_result()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("low", low)
@pytest.mark.parametrize("high", high)
def test_randint_low(low, high, size):
    input_params = {
            'low': low,
            'high': high,
            'size': size,
            'device': 'musa'}
    test = testing.OpTest(
        func=torch.randint,
        input_args=input_params,
        ignored_result_indices=[0]
    )
    test.check_result()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("high", high)
def test_randint(high, size):
    input_params = {
            'high': high,
            'size': size,
            'device': 'musa'}
    test = testing.OpTest(
        func=torch.randint,
        input_args=input_params,
        ignored_result_indices=[0]
    )
    test.check_result()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("high", high)
@pytest.mark.parametrize("generator", generator)
def test_randint_generator(high, size, generator):
    input_params = {
            'high': high,
            'size': size,
            'generator' : generator,
            'device': 'musa'}
    test = testing.OpTest(
        func=torch.randint,
        input_args=input_params,
        ignored_result_indices=[0]
    )
    test.check_result()

start = [0, 5]
end = [10, 20]
step = [1, 2]
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("start", start)
@pytest.mark.parametrize("end", end)
@pytest.mark.parametrize("step", step)
def test_arange(start, end, step):
    input_params = {
            'start': start,
            'end': end,
            'step' : step}
    test = testing.OpTest(
        func=torch.arange,
        input_args=input_params,
    )
    test.check_result()
