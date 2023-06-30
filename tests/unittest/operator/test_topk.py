"""Test topk operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = testing.get_raw_data()
# dtype of input tensor of topk only support Float32 in muDNN now.
support_dtypes = [torch.float32]
largest = [False, True]

# TODO(MT-AI): fix error when testing on GPU 1
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("largest", largest)
@pytest.mark.parametrize("sort", [True])
def test_topk_sorted_true(input_data, dtype, largest, sort):
    input_args = {}
    input_args["input"] = input_data.to(dtype)
    dim = torch.randint(input_data.dim(), (1, )).item()
    k = torch.randint(input_data.size()[dim], (1, )).item()
    input_args["dim"] = dim
    input_args["k"] = k
    input_args["largest"] = largest
    input_args["sorted"] = sort
    test = testing.OpTest(func=torch.topk, input_args=input_args)
    test.check_result()

# TODO(MT-AI): fix error when testing on GPU 1
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("largest", largest)
@pytest.mark.parametrize("sort", [False])
@pytest.mark.parametrize("k", [1, 2])
def test_topk_sorted_false(input_data, dtype, largest, sort, k):
    input_args = {}
    input_args["input"] = input_data.to(dtype)
    input_args["dim"] = 0
    input_args["k"] = k
    input_args["largest"] = largest
    input_args["sorted"] = sort
    test = testing.OpTest(func=torch.topk,
                          input_args=input_args,
                          ignored_result_indices=[1])
    test.check_result()
