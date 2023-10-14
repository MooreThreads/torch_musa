"""Test index_put operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random
import numpy as np
import pytest
import torch
import torch_musa
from torch_musa import testing


def get_indices(inputs):
    indices = []
    values = []
    for input_self in inputs:
        indice = []  # tuple of LongTensor, and its length must match the input's dim
        # the number of items must not exceed input items
        num_item = min(10, input_self.numel())
        shape = input_self.shape
        for dim in tuple(range(input_self.dim())):
            max_idx = shape[dim]
            indice.append(torch.randint(max_idx, (num_item,)))

        indices.append(tuple(indice))
        values.append(torch.randn(num_item))
    return [indices, values]


input_data = testing.get_raw_data()

[indices, values] = get_indices(input_data)

input_datas = []
for i, data in enumerate(input_data):
    input_datas.append({"input": data, "indices": indices[i], "values": values[i]})

# muDNN index_put ot only support torch.float32 now
dtypes = [torch.float32]


# test index_input
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_index_put(input_data, dtype):
    input_data["input"] = input_data["input"].to(dtype)
    input_data["values"] = input_data["values"].to(dtype)
    test = testing.OpTest(func=torch.index_put, input_args=input_data)
    test.check_result()


tensor_dtype = [torch.int32, torch.int64, torch.float32, torch.float64]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("tensor_dtype", tensor_dtype)
def test_index_put_different_device_indices(tensor_dtype):
    input_data = torch.randperm(20, dtype=tensor_dtype).reshape(1, 20)
    value = torch.randperm(5, dtype=tensor_dtype)
    indices_0_cpu = torch.tensor([[0]], device="cpu")
    indices_1_musa = torch.tensor([[3, 2, 0, 17, 19]], device="musa")

    input_musa = input_data.to("musa")
    target_device = input_musa.device
    input_musa[indices_0_cpu, indices_1_musa] = value.to("musa")
    musa_result = input_musa.cpu()

    input_cpu_golden = input_data.clone()
    indices_1_cpu = torch.tensor([[3, 2, 0, 17, 19]], device="cpu")
    input_cpu_golden[indices_0_cpu, indices_1_cpu] = value
    cpu_result = input_cpu_golden

    assert (musa_result == cpu_result).all()
    assert musa_result.dtype == cpu_result.dtype
    assert musa_result.shape == cpu_result.shape
    assert input_musa.device == target_device


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("tensor_dtype", tensor_dtype)
def test_index_tensor(tensor_dtype):
    input_data = torch.randperm(40, dtype=tensor_dtype).reshape(2, 20)
    indices_0_musa = torch.tensor([[0], [1]], device="musa")
    indices_1_musa = torch.tensor([[3, 2, 0, 17, 19],
                                   [19, 17, 0, 2, 3]], device="musa")

    input_musa = input_data.to("musa")
    musa_result = input_musa[indices_0_musa, indices_1_musa]

    input_cpu_golden = input_data.clone()
    indices_0_cpu = torch.tensor([[0], [1]], device="cpu")
    indices_1_cpu = torch.tensor([[3, 2, 0, 17, 19],
                                   [19, 17, 0, 2, 3]], device="cpu")
    cpu_result = input_cpu_golden[indices_0_cpu, indices_1_cpu]

    assert (musa_result.cpu() == cpu_result).all()
    assert musa_result.dtype == cpu_result.dtype
    assert musa_result.shape == cpu_result.shape
    assert musa_result.device == input_musa.device


# test index_select
input_datas = [
    {
        "input": torch.zeros(
            10,
        ),
        "dim": 0,
        "index": torch.randint(10, (5,)),
    },
    {"input": torch.zeros(10, 5), "dim": 1, "index": torch.randint(5, (3,))},
    {"input": torch.zeros(10, 5, 3), "dim": 2, "index": torch.randint(3, (2,))},
    {"input": torch.zeros(10, 5, 1, 3), "dim": 1, "index": torch.randint(1, (1,))},
    {"input": torch.zeros(10, 5, 1, 3, 5), "dim": 4, "index": torch.randint(5, (3,))},
    {
        "input": torch.zeros(10, 5, 1, 3, 2, 6),
        "dim": 1,
        "index": torch.randint(5, (3,)),
    },
    {
        "input": torch.zeros(10, 5, 1, 3, 1, 2, 7),
        "dim": 3,
        "index": torch.randint(3, (2,)),
    },
    {
        "input": torch.zeros(10, 5, 1, 3, 1, 2, 3, 8),
        "dim": 7,
        "index": torch.randint(8, (3,)),
    },
]
dtypes = testing.get_all_support_types()
dtypes.extend([torch.uint8, torch.int16, torch.float16])


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_index_select(input_data, dtype):
    input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(func=torch.index_select, input_args=input_data)
    test.check_result()
