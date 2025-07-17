"""Test heaviside operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import copy
import torch
import pytest

from torch_musa import testing

shapes = [[(1, 10), (1,)], [(10, 20), (1, 20)], [(4, 5), (4, 5)]]

dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.float64]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_heaviside(shape, dtype):
    inputs = {}
    inputs["input"] = torch.randn(shape[0]).to(dtype)
    inputs["values"] = torch.randn(shape[1]).to(dtype)
    test = testing.OpTest(
        func=torch.heaviside,
        input_args=inputs,
    )
    test.check_result()
    test.check_out_ops()

    inplace_input = copy.deepcopy(inputs)
    self_tensor = inplace_input["input"]
    inplace_input.pop("input")
    test = testing.InplaceOpChek(
        func_name="heaviside",
        self_tensor=self_tensor,
        input_args=inplace_input,
    )
    test.check_address()
    test.check_res()
