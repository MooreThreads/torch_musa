"""Test cat operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, C0103
import torch
import pytest
import torch_musa
from torch_musa import testing

inputdata = [
    {
        "input": [torch.randn(1), torch.randn(1)],
        "dim": 0,
    },
    {
        "input": [torch.randn(1, 2), torch.randn(1, 2)],
        "dim": 1,
    },
    {
        "input": [torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)],
        "dim": 2,
    },
    {
        "input": [torch.randn(1, 2, 3, 0), torch.randn(1, 2, 3, 0)],
        "dim": 2,
    },
    {
        "input": [torch.randn(1, 0, 3, 4), torch.randn(1, 0, 3, 4)],
        "dim": 2,
    },
    {
        "input": [torch.randn(1, 0, 0, 4), torch.randn(1, 0, 0, 4)],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(2, 1, 2, 3).to(memory_format=torch.channels_last),
            torch.randn(2, 1, 2, 3).to(memory_format=torch.channels_last),
        ],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(2, 1, 0, 3).to(memory_format=torch.channels_last),
            torch.randn(2, 1, 0, 3).to(memory_format=torch.channels_last),
        ],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(2, 0, 2, 3).to(memory_format=torch.channels_last),
            torch.randn(2, 0, 2, 3).to(memory_format=torch.channels_last),
        ],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(2, 3, 1, 1).to(memory_format=torch.channels_last),
            torch.randn(2, 3, 1, 1).to(memory_format=torch.channels_last),
        ],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(2, 3, 6, 4).to(memory_format=torch.channels_last),
            torch.randn(2, 3, 6, 4).to(memory_format=torch.channels_last),
        ],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(2, 3, 1, 1).to(memory_format=torch.channels_last),
            torch.randn(2, 3, 1, 1),
        ],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(2, 3, 1, 1),
            torch.randn(2, 3, 1, 1).to(memory_format=torch.channels_last),
        ],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
        ],
        "dim": 3,
    },
    {
        "input": [
            torch.randn(1, 2, 3, 4),
            torch.tensor([]),
            torch.randn(1, 2, 3, 4),
        ],
        "dim": 3,
    },
    {
        "input": [
            torch.tensor([]),
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
        ],
        "dim": -2,
    },
    {
        "input": [
            torch.tensor([]),
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 1, 4),
            torch.randn(1, 2, 10, 4),
        ],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(1, 2, 3, 4)[:, :, :, :0],
            torch.randn(1, 2, 3, 4),
        ],
        "dim": -1,
    },
    {
        "input": [
            torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)[:, :0, :, :],
            torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),
        ],
        "dim": 1,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inputdata)
def test_cat(input_data):
    inputs = {"tensors": input_data["input"], "dim": input_data["dim"]}
    test = testing.OpTest(
        func=torch.cat,
        input_args=inputs,
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inputdata)
def test_cat_with_nhwc(input_data):
    data = input_data["input"]
    if data[0].dim == 4:
        data = [x.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1) for x in data]
    inputs = {"tensors": data, "dim": input_data["dim"]}
    test = testing.OpTest(
        func=torch.cat,
        input_args=inputs,
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "dtype", testing.get_all_types() + [torch.half, torch.int8, torch.int16]
)
def test_cat_zero_shape(dtype):
    x0 = torch.randn(1000, 4).to(dtype)
    x1 = torch.randn(0, 4).to(dtype)
    y_cpu = torch.cat([x0, x1], dim=0)
    y_musa = torch.cat([x0.to("musa"), x1.to("musa")], dim=0)
    testing.DefaultComparator(y_musa, y_cpu)


inputdata = [
    {
        "input": [torch.randn(1), torch.randn(1).half()],
        "dim": 0,
    },
    {
        "input": [torch.randn(1, 2).half(), torch.randn(1, 2)],
        "dim": 1,
    },
    {
        "input": [torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4).half()],
        "dim": 2,
    },
    {
        "input": [
            torch.randn(1, 2, 3, 4).half(),
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4).half(),
        ],
        "dim": 3,
    },
    {
        "input": [
            torch.randn(1, 0, 3, 4).half(),
            torch.randn(1, 0, 3, 4),
            torch.randn(1, 0, 3, 4).half(),
        ],
        "dim": 3,
    },
    {
        "input": [
            torch.randn(1, 2, 0, 4).half(),
            torch.randn(1, 2, 0, 4),
            torch.randn(1, 2, 0, 4).half(),
        ],
        "dim": 3,
    },
    {
        "input": [
            torch.randn(1, 2, 3, 4).half(),
            torch.tensor([]),
            torch.randn(1, 2, 3, 4).half(),
        ],
        "dim": 3,
    },
    {
        "input": [
            torch.tensor([]),
            torch.randn(1, 2, 3, 4).half(),
            torch.randn(1, 2, 3, 4),
        ],
        "dim": -2,
    },
    {
        "input": [
            torch.tensor([]),
            torch.randn(1, 2, 3, 4).half(),
            torch.randn(1, 2, 1, 4),
            torch.randn(1, 2, 10, 4).half(),
        ],
        "dim": 2,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inputdata)
def test_cat_with_different_dtype(input_data):
    inputs = {"tensors": input_data["input"], "dim": input_data["dim"]}
    test = testing.OpTest(
        func=torch.cat,
        input_args=inputs,
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


inputdata = [
    {
        "input": [
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
        ],
        "dim": 4,
    },
    {
        "input": [
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
        ],
        "dim": 1,
    },
    {
        "input": [
            torch.randn(1, 2, 2, 2, 2),
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
            torch.randn(1, 2, 2, 2, 2),
        ],
        "dim": 4,
    },
    {
        "input": [
            torch.randn(1, 2, 2, 2, 2),
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
            torch.randn(1, 2, 2, 2, 2),
        ],
        "dim": 1,
    },
    {
        "input": [
            torch.tensor([]),
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
            torch.randn(1, 2, 2, 2, 2),
        ],
        "dim": -2,
    },
    {
        "input": [
            torch.tensor([]),
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
            torch.randn(1, 2, 2, 2, 2),
            torch.randn(1, 2, 2, 2, 2).to(memory_format=torch.channels_last_3d),
        ],
        "dim": 2,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inputdata)
def test_cat_ndhwc(input_data):
    inputs = {"tensors": input_data["input"], "dim": input_data["dim"]}
    test = testing.OpTest(
        func=torch.cat,
        input_args=inputs,
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
# only fp32, int8 dtype will be fine
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_cat_zero_numel_multi_dim(dtype):
    tensor_lst = [torch.randn(0, 2048, dtype=dtype) for i in range(6)]
    tensor_lst_musa = [t.to(torch.musa.current_device()) for t in tensor_lst]
    out_cpu = torch.cat(tensor_lst, dim=1)
    out_musa = torch.cat(tensor_lst_musa, dim=1)

    assert out_cpu.shape == out_musa.shape
    assert out_cpu.dtype == out_musa.dtype


# ============================ complex ============================
complex_full_inputdata = [
    {
        "input": [
            torch.complex(torch.randn(2, 3), torch.randn(2, 3)),
            torch.complex(torch.randn(2, 3), torch.randn(2, 3)),
        ],
        "dim": 1,
    },
    {
        "input": [1.0 + 1j * torch.randn(2, 3), 2.0 + 1j * torch.randn(2, 3)],
        "dim": 1,
    },
    {
        "input": [
            torch.randn(2, 3) + 1j * torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 3) * 1j,
        ],
        "dim": 0,
    },
    {
        "input": [
            torch.tensor([1 + 1j, 2 + 2j, 3 + 3j]),
            torch.tensor([4 + 4j, 5 + 5j, 6 + 6j]),
        ],
        "dim": 0,
    },
    {
        "input": [
            torch.randn(1, 0, 3) + 1j * torch.randn(1, 0, 3),
            torch.randn(1, 0, 3) + 1j * torch.randn(1, 0, 3),
        ],
        "dim": 2,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", complex_full_inputdata)
def test_cat_complex(input_data):
    inputs = {"tensors": input_data["input"], "dim": input_data["dim"]}
    test = testing.OpTest(
        func=torch.cat,
        input_args=inputs,
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()
