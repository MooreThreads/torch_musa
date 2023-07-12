"""Test reduce operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
from torch_musa import testing

input_data = [
        {"input": torch.randn([1, 10]), "dim": 1},
        {"input": torch.randn([1, 10, 5]), "dim": 2},
        {"input": torch.randn([1, 10, 5, 5]), "dim": 3},
        {"input": torch.randn([1, 10, 5, 5, 10]), "dim": 4},
        {"input": torch.randn([9, 8, 7, 6, 5, 4]), "dim": 5},
        {"input": torch.randn([9, 8, 7, 6, 5, 4, 16]), "dim": 5},
        {"input": torch.randn([9, 8, 7, 6, 5, 4, 5, 20]), "dim": 7}
]

def function(input_data, dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(func=func,
                          input_args=input_data,
                          comparators=testing.DefaultComparator(abs_diff=1e-5))
    test.check_result()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_amax(input_data, dtype):
    function(input_data, dtype, torch.amax)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_mean(input_data, dtype):
    function(input_data, dtype, torch.mean)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sum(input_data, dtype):
    function(input_data, dtype, torch.sum)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_prod(input_data, dtype):
    function(input_data, dtype, torch.prod)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_norm(input_data, dtype):
    function(input_data, dtype, torch.norm)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cumsum(input_data, dtype):
    function(input_data, dtype, torch.cumsum)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.bool])
def test_any(input_data, dtype):
    function(input_data, dtype, torch.any)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_max(input_data, dtype):
    function(input_data, dtype, torch.max)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_min(input_data, dtype):
    function(input_data, dtype, torch.min)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.bool])
def test_all(input_data, dtype):
    function(input_data, dtype, torch.all)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_argmax(input_data, dtype):
    function(input_data, dtype, torch.argmax)
