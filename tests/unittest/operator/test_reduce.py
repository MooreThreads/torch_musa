"""Test reduce operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import numpy as np
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


extra_data_for_cumsum = [
        {"input": torch.rand(3, 4) < 0.5, "dim": 1},
        {"input": torch.rand([1, 10, 5]) < 0.5, "dim": 2},
        {"input": torch.rand([1, 10, 5, 5]) < 0.5, "dim": 3},
        {"input": torch.rand([1, 10, 5, 5, 10]) < 0.5, "dim": 4},
        {"input": torch.rand([9, 8, 7, 6, 5, 4]) < 0.5, "dim": 5},
        {"input": torch.rand([9, 8, 7, 6, 5, 4, 16]) < 0.5, "dim": 5},
        {"input": torch.rand([9, 8, 7, 6, 5, 4, 5, 20]) < 0.5, "dim": 7}
]
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data+extra_data_for_cumsum)
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


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", [
    [(0,), [0], ],
    [(450,), [0], ],
    [(2111, 3000), [1],],
    [(4, 5, 6, 7), [0, 2], ],
    [(2, 3, 4, 5, 6, 7), [1, 3, 5], ],
    [(2, 3, 4, 5, 6, 7, 8, 9), [0, 2, 4, 6], ],
])
def test_sum_i32_in_f32_out(config):
    min_val, max_val = -5, 5
    x_np = np.random.uniform(min_val, max_val, size=config[0]).astype("int32")
    x_tensor = torch.from_numpy(x_np)
    test = testing.OpTest(func=torch.sum,
                          input_args={"input": x_tensor,
                                      "dim": config[1],
                                      "dtype": torch.float32},
                          comparators=testing.DefaultComparator(abs_diff=1e-8))
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", [
    [(5,), 0, ],
    [(2, 3), 1, ],
    [(5, 3, 2), 1, ],
    [(2, 2, 5, 2), 3, ],
    [(2, 2, 1, 1, 1, 1), 5, ]
])
@pytest.mark.parametrize("interval", [[-5, 5], [1, 5]])
def test_prod_i32_in_f32_out(config, interval):
    min_val, max_val = interval[0], interval[1]
    x_np = np.random.uniform(min_val, max_val, size=config[0]).astype("int32")
    x_tensor = torch.from_numpy(x_np)
    test = testing.OpTest(func=torch.prod,
                          input_args={"input": x_tensor,
                                      "dim": config[1],
                                      "dtype": torch.float32},
                          comparators=testing.DefaultComparator(abs_diff=1e-8))
    test.check_result()
