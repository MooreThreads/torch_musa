"""Test unary complex operators."""

from typing import List
import torch
import pytest

from torch_musa import testing


angle_shapes = [
    [
        128,
    ],
    [16, 128],
    [2, 16, 128],
    [2, 4, 16, 128],
]
angle_supported_dtypes = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("shape", angle_shapes)
@pytest.mark.parametrize("dtype", angle_supported_dtypes)
@pytest.mark.parametrize("test_out", [False, True])
def test_angle(shape: List, dtype: torch.dtype, test_out: bool) -> None:
    """test angle/angle.out"""
    if dtype == torch.complex32:
        real = torch.empty(shape, dtype=torch.float16).uniform_(1, 2)
        img = torch.empty(shape, dtype=torch.float16).uniform_(1, 2)
        input_tensor = torch.complex(real, img)
        if test_out:
            output_tensor = torch.empty(shape, dtype=torch.float16)
    elif dtype == torch.complex64:
        real = torch.empty(shape, dtype=torch.float32).uniform_(1, 2)
        img = torch.empty(shape, dtype=torch.float32).uniform_(1, 2)
        input_tensor = torch.complex(real, img)
        if test_out:
            output_tensor = torch.empty(shape, dtype=torch.float32)
    elif dtype == torch.complex128:
        real = torch.empty(shape, dtype=torch.float64).uniform_(1, 2)
        img = torch.empty(shape, dtype=torch.float64).uniform_(1, 2)
        input_tensor = torch.complex(real, img)
        if test_out:
            output_tensor = torch.empty(shape, dtype=torch.float64)
    else:
        input_tensor = torch.empty(shape).uniform_(1, 2).to(dtype)
        if test_out:
            output_tensor = torch.empty_like(input_tensor)

    input_args = {"input": input_tensor}
    if test_out:
        input_args["out"] = output_tensor

    op_test = testing.OpTest(func=torch.angle, input_args=input_args)
    op_test.check_result(test_out=test_out)
