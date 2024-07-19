"""Test range operators."""

# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing
import torch_musa

start = [0, 2, 4, 6, 8]
end = [10, 20, 30, 40, 50]
step = [1, 2, 3, 4]
dtype = [torch.float, torch.double, torch.int, torch.long]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("start", start)
@pytest.mark.parametrize("end", end)
@pytest.mark.parametrize("step", step)
@pytest.mark.parametrize("dtype", dtype)
def test_arange(start, end, step, dtype):
    input_params = {"start": start, "end": end, "step": step, "dtype": dtype}
    test = testing.OpTest(
        func=torch.range,
        input_args=input_params,
    )
    test.check_result()
    test.check_out_ops()


start_negstep = [10, 20, 30, 40, 50]
end_negstep = [0, 2, 4, 6, 8]
step_negstep = [-1, -2, -3, -4]
dtype_negstep = [torch.float, torch.double, torch.int, torch.long]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("start", start_negstep)
@pytest.mark.parametrize("end", end_negstep)
@pytest.mark.parametrize("step", step_negstep)
@pytest.mark.parametrize("dtype", dtype_negstep)
def test_arange_negstep(start, end, step, dtype):
    input_params = {"start": start, "end": end, "step": step, "dtype": dtype}
    test = testing.OpTest(
        func=torch.range,
        input_args=input_params,
    )
    test.check_result()
    test.check_out_ops()
