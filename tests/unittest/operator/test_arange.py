"""Test arange operators."""
# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing
import torch_musa

start = [0, 5]
end = [10, 20]
step = [1, 2]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("start", start)
@pytest.mark.parametrize("end", end)
@pytest.mark.parametrize("step", step)
def test_arange(start, end, step):
    input_params = {"start": start, "end": end, "step": step}
    test = testing.OpTest(
        func=torch.arange,
        input_args=input_params,
    )
    test.check_result()
