"""Test binary operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
import numpy as np
import torch_musa
from torch_musa import testing

data_type = [torch.float32, torch.int32, torch.int64, torch.float64]

input_data = [
    {"input": np.array(5.0)},
    {"input": np.random.randn(1)},
    {"input": np.random.randn(1, 2)},
    {"input": np.random.randn(1, 2, 3)},
    {"input": np.random.randn(1, 2, 3)},
    {"input": np.random.randn(1, 0, 3)},
    {"input": np.random.randn(1, 2, 3, 4)},
    {"input": np.random.randn(1, 2, 3, 4, 3)},
    {"input": np.random.randn(1, 2, 3, 4, 3, 2)},
    {"input": np.random.randn(1, 2, 3, 4, 3, 2, 4)},
    {"input": np.random.randn(1, 2, 3, 4, 3, 2, 4, 2)},
]


@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_tensor_new(input_data, data_type):
    test = testing.OpTest(func=torch.Tensor.new, input_args={})
    test.check_result(torch.tensor(data=input_data["input"], dtype=data_type))


@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_tensor_a_new(input_data, data_type):
    mtgpu_tensor = torch.tensor(data=input_data["input"], dtype=data_type, device="musa")
    new_mtgpu_result = mtgpu_tensor.new(input_data["input"])

    cpu_tensor = torch.tensor(data=input_data["input"], dtype=data_type, device="cpu")
    new_cpu_result = cpu_tensor.new(input_data["input"])

    assert new_mtgpu_result.shape == new_cpu_result.shape
    assert new_mtgpu_result.dtype == new_cpu_result.dtype

    if testing.MULTIGPU_AVAILABLE:
        with torch.musa.device(1):
            mtgpu_tensor = torch.tensor(data=input_data["input"], dtype=data_type, device="musa")
            new_mtgpu_result = mtgpu_tensor.new(input_data["input"])

            assert new_mtgpu_result.shape == new_cpu_result.shape
            assert new_mtgpu_result.dtype == new_cpu_result.dtype


@testing.skip_if_not_multiple_musa_device
def test_new():
    x = torch.randn(3, 3).to("musa")
    assert x.new([0, 1, 2]).get_device() == 0
    assert x.new([0, 1, 2], device="musa:1").get_device() == 1

    with torch.musa.device(1):
        assert x.new([0, 1, 2]).get_device() == 0
        assert x.new([0, 1, 2], device="musa:1").get_device() == 1


def test_scalar_tensortype():
    dtype_dict = {
        np.bool_: [torch.musa.BoolTensor, torch.BoolTensor],
        np.int8: [torch.musa.CharTensor, torch.CharTensor],
        np.uint8: [torch.musa.ByteTensor, torch.ByteTensor],
        np.float16: [torch.musa.HalfTensor, torch.HalfTensor],
        np.int16: [torch.musa.ShortTensor, torch.ShortTensor],
        np.int32: [torch.musa.IntTensor, torch.IntTensor],
        np.int64: [torch.musa.LongTensor, torch.LongTensor],
        np.float32: [torch.musa.FloatTensor, torch.FloatTensor],
        np.float64: [torch.musa.DoubleTensor, torch.DoubleTensor],
    }

    for data_type, tensor_type in dtype_dict.items():
        data = np.random.randn(3, 5).astype(data_type)
        musa_tensor = tensor_type[0](data)
        np_tensor = tensor_type[1](data)
        assert musa_tensor.dtype == np_tensor.dtype
        assert musa_tensor.device.type == "musa"
        np.testing.assert_allclose(musa_tensor.to("cpu"), np_tensor)


def test_tensor_type():
    tensor = torch.rand(3, 3, device="musa:0")
    assert tensor.is_musa

    for tensor in torch._tensor_classes:
        if "musa" in tensor.__module__:
            assert tensor.is_musa
        else:
            assert not ("musa" in tensor.__module__ and tensor.is_musa)


def test_set_dtype():
    dtype_dict = {
        torch.musa.BoolTensor : "torch.musa.BoolTensor",
        torch.musa.CharTensor : "torch.musa.CharTensor",
        torch.musa.ByteTensor : "torch.musa.ByteTensor",
        torch.musa.HalfTensor : "torch.musa.HalfTensor",
        torch.musa.FloatTensor : "torch.musa.FloatTensor",
        torch.musa.DoubleTensor : "torch.musa.DoubleTensor",
    }

    for dtype, type_str in dtype_dict.items():
        x = torch.randn(3, 3)
        y = x.type(dtype)
        assert x.device == torch.device('cpu')
        assert y.type() == type_str
        assert y.device == torch.device('musa:0')
