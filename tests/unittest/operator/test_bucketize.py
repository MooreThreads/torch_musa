"""Test bucketize operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = testing.get_raw_data()
# dtype of input tensor of bucketize only support Float32 in muDNN now.
support_dtypes = [torch.float32]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_bucketize(input_data, dtype):
    input_args = {}
    input_args["input"] = input_data.to(dtype)
    input_args["boundaries"] = torch.tensor([2.0, 4.0])
    test = testing.OpTest(func=torch.bucketize, input_args=input_args)
    test.check_result()
