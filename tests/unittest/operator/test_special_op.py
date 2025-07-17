"""Test torch.special.* operators on MUSA """

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    {
        "input": torch.rand(0),
    },
    {
        "input": torch.rand(5),
    },
    {
        "input": torch.rand(4, 0),
    },
    {
        "input": torch.rand(10, 10),
    },
    {
        "input": torch.rand(2, 256),
    },
    {
        "input": torch.rand(16, 32, 8),
    },
]

float_dtypes = [torch.float32, torch.float16]
# bf16 is not supported on arch older than qy2
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_i0(input_data):
    test = testing.OpTest(
        func=torch.i0,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_i0_(input_data):
    test = testing.InplaceOpChek(
        func_name=torch.i0.__name__ + "_",
        self_tensor=input_data["input"],
        comparators=[testing.DefaultComparator(abs_diff=1e-3)],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("float_dtype", float_dtypes)
def test_i0_out(input_data, float_dtype):
    input_tensor = input_data["input"].clone().to(float_dtype)
    data = {"input": input_tensor, "out": torch.zeros_like(input_tensor)}
    test = testing.OpTest(
        func=torch.i0,
        input_args=data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
