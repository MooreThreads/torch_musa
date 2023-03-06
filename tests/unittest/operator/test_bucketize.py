"""Test bucketize operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = testing.get_raw_data()
support_dtypes = [torch.float32, torch.int64, torch.int32]
rights = [True, False]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("right", rights)
def test_bucketize(input_data, dtype, right):
    input_args = {}
    input_args["input"] = input_data.to(dtype)
    input_args["boundaries"] = torch.tensor([2.0, 4.0])
    input_args["right"] = right
    test = testing.OpTest(func=torch.bucketize, input_args=input_args)
    test.check_result()
