"""Test onehot operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_shapes = [
    [20],
    [5, 10],
    [8, 32],
    [8, 16, 2],
    [16, 16, 2],
    [8, 32, 8],
    [16, 32, 32],
    [32, 32, 32],
]
support_dtypes = [
    torch.int64,
]
num_classes = [-1, 10, 20]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("num_classes", num_classes)
def test_onehot(input_shape, dtype, num_classes):
    upper = num_classes if num_classes > 0 else 20
    input_args = {}
    input_args["input"] = torch.randint(0, upper, input_shape, dtype=dtype)
    input_args["num_classes"] = num_classes
    test = testing.OpTest(func=torch.nn.functional.one_hot, input_args=input_args)
    test.check_result()
