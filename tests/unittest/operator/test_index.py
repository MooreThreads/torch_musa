"""Test index_put operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
import torch_musa
from torch_musa import testing

torch.manual_seed(41)


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

# index_put ot only support torch.float16/32 and int64 now
dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.int32, torch.int64]

ind_dtypes = [torch.int64]  # cpu only support int64 indices


# test index_put
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ind_dtype", ind_dtypes)
def test_index_put(input_data, dtype, ind_dtype):
    input_data["input"] = input_data["input"].to(dtype)
    input_data["indices"] = [x.to(ind_dtype) for x in input_data["indices"]]
    input_data["values"] = input_data["values"].to(dtype)
    test = testing.OpTest(func=torch.index_put, input_args=input_data)
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("tensor_dtype", dtypes)
@pytest.mark.parametrize("ind_dtype", ind_dtypes)
def test_index_put_different_device_indices(tensor_dtype, ind_dtype):
    input_data = torch.randperm(20, dtype=tensor_dtype).reshape(1, 20)
    value = torch.randperm(5, dtype=tensor_dtype)
    indices_0_cpu = torch.tensor([[0]], device="cpu", dtype=ind_dtype)
    indices_1_musa = torch.tensor([[3, 2, 0, 17, 19]], device="musa", dtype=ind_dtype)

    input_musa = input_data.to("musa")
    target_device = input_musa.device
    input_musa[indices_0_cpu, indices_1_musa] = value.to("musa")
    musa_result = input_musa.cpu()

    input_cpu_golden = input_data.clone()
    indices_1_cpu = torch.tensor([[3, 2, 0, 17, 19]], device="cpu", dtype=ind_dtype)
    input_cpu_golden[indices_0_cpu, indices_1_cpu] = value
    cpu_result = input_cpu_golden

    assert (musa_result == cpu_result).all()
    assert musa_result.dtype == cpu_result.dtype
    assert musa_result.shape == cpu_result.shape
    assert input_musa.device == target_device


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_index_put_bool_index(input_data, dtype):
    data = input_data["input"].to(dtype)
    inds = torch.randn(data.shape)
    inds = inds.ge(0)
    if inds.sum() == 0:  # for random inds are all less than 0.
        return
    musa_data = data.musa()

    data[inds] = 1.0
    musa_data[inds] = 1.0
    assert torch.allclose(data, musa_data.cpu())


# test index
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("tensor_dtype", dtypes)
@pytest.mark.parametrize("ind_dtype", ind_dtypes)
def test_index_tensor(tensor_dtype, ind_dtype):
    input_data = torch.randperm(40, dtype=tensor_dtype).reshape(2, 20)
    indices_0_musa = torch.tensor([[0], [1]], device="musa", dtype=ind_dtype)
    indices_1_musa = torch.tensor(
        [[3, 2, 0, 17, 19], [19, 17, 0, 2, 3]], device="musa", dtype=ind_dtype
    )

    input_musa = input_data.to("musa")
    musa_result = input_musa[indices_0_musa, indices_1_musa]

    input_cpu_golden = input_data.clone()
    indices_0_cpu = torch.tensor([[0], [1]], device="cpu", dtype=ind_dtype)
    indices_1_cpu = torch.tensor(
        [[3, 2, 0, 17, 19], [19, 17, 0, 2, 3]], device="cpu", dtype=ind_dtype
    )
    cpu_result = input_cpu_golden[indices_0_cpu, indices_1_cpu]

    assert (musa_result.cpu() == cpu_result).all()
    assert musa_result.dtype == cpu_result.dtype
    assert musa_result.shape == cpu_result.shape
    assert musa_result.device == input_musa.device


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("tensor_dtype", dtypes)
def test_index_tensor_bool_index(tensor_dtype):
    input_data = torch.randperm(40, dtype=tensor_dtype).reshape(2, 20)
    inds = torch.randn(input_data.shape)
    inds = inds.ge(0.5)
    input_musa = input_data.to("musa")
    musa_result = input_musa[inds]

    input_cpu_golden = input_data.clone()
    cpu_result = input_cpu_golden[inds]

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
