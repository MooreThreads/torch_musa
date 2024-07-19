"""Test nonzero operators."""

# pylint: disable=missing-function-docstring
import torch
import pytest

from torch_musa import testing

support_dtypes = testing.get_all_types() + [
    torch.bfloat16,
]
input_shapes = [
    (451143,),
    (10,),
    (256, 2),
    (2, 4, 6, 8),
    (2, 2, 4, 5, 8),
    (4, 8, 8, 2, 2, 8),
    (4, 8, 8, 2, 2, 16, 2),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("as_tuple", [False, True])
def test_nonzero(input_shape, dtype, as_tuple):
    if testing.get_musa_arch() < 22 and dtype == torch.bfloat16:
        return
    input_args = {}
    input_args["input"] = torch.randn(input_shape).to(dtype)
    input_args["as_tuple"] = as_tuple
    test = testing.OpTest(func=torch.nonzero, input_args=input_args)
    test.check_result()
    test.check_grad_fn()


input_datas = [
    ((451143,), 0),
    ((10,), -1),
    ((256, 2), 1),
    ((2, 4, 6, 8),),
    ((2, 2, 4, 5, 8), 3),
    ((4, 8, 8, 2, 2, 8), 1),
    ((4, 8, 8, 2, 2, 16, 2), (2, 3, 5)),
]
support_dtypes.remove(torch.bool)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_count_nonzero(input_data, dtype):
    if testing.get_musa_arch() < 22 and dtype == torch.bfloat16:
        return
    input_shape = input_data[0]
    input_tensor = torch.randint(0, 10, input_shape, dtype=dtype)
    dim = input_data[1] if len(input_data) > 1 else 0
    input_args = {"input": input_tensor, "dim": dim}
    test = testing.OpTest(func=torch.count_nonzero, input_args=input_args)
    test.check_result()
    test.check_grad_fn()
