"""Test tensor_factory operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = testing.get_raw_data()
input_data.append(torch.tensor(random.uniform(-10, 10)))
support_dtypes = testing.get_all_support_types()


def _check_result(musa_res, cpu_res):
    compare = testing.DefaultComparator
    assert musa_res.dtype == cpu_res.dtype
    assert musa_res.shape == cpu_res.shape
    assert compare(musa_res, cpu_res)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_set_(input_data, dtype):
    input_data = input_data.to(dtype)
    musa_input = input_data.clone().to("musa")
    musa_res = musa_input.set_()
    cpu_res = input_data.set_()

    _check_result(musa_res, cpu_res)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_set_source_tensor(input_data, dtype):
    source_data = torch.rand(3, 5).to(dtype)
    input_data = input_data.to(dtype)
    musa_input = input_data.clone().to("musa")
    musa_res = musa_input.set_(source_data.to("musa"))
    cpu_res = input_data.set_(source_data)

    _check_result(musa_res, cpu_res)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_set_source_storage(input_data, dtype):
    source_data = torch.rand(3, 5).to(dtype)
    input_data = input_data.to(dtype)
    musa_input = input_data.clone().to("musa")
    musa_res = musa_input.set_(source_data.to("musa").untyped_storage())
    cpu_res = input_data.set_(source_data.untyped_storage())

    _check_result(musa_res, cpu_res)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_resize_(input_data, dtype):
    input_data = input_data.to(dtype)
    size = (3, 4)
    musa_input = input_data.clone().to("musa")
    musa_res = musa_input.resize_(size)
    cpu_res = input_data.resize_(size)

    _check_result(musa_res, cpu_res)


size_strides = [
    [(1,), (1,)],
    [(2, 3), (3, 2)],
    [(2, 3, 4), (3, 2, 1)],
    [(2, 3, 4, 5), (3, 2, 1, 2)],
    [(2, 3, 4, 5, 6), (3, 2, 1, 2, 3)],
    [(2, 3, 1, 2, 2, 7), (3, 2, 1, 2, 3, 4)],
    [(2, 3, 1, 2, 1, 2, 8), (3, 2, 1, 2, 1, 2, 5)],
    [(2, 3, 1, 2, 1, 2, 2, 9), (3, 2, 1, 2, 1, 2, 3, 2)],
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@ pytest.mark.parametrize("size_strides", size_strides)
@ pytest.mark.parametrize("dtype", support_dtypes)
def test_empty_strided(size_strides, dtype):
    size, stride = size_strides
    musa_res = torch.empty_strided(size, stride, dtype=dtype, device="musa")
    cpu_res = torch.empty_strided(size, stride, dtype=dtype, device="cpu")
    _check_result(musa_res, cpu_res)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("m", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_eye(n, m, dtype):
    musa_res = torch.eye(n, m, dtype=dtype, device="musa")
    cpu_res = torch.eye(n, m, dtype=dtype, device="cpu")
    _check_result(musa_res, cpu_res)
