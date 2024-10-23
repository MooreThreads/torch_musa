"""Test unique operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

torch.random.manual_seed(41)

input_data = [
    {"input": torch.tensor(1.32)},
    {"input": torch.randn([1, 8, 4])},
    {
        "input": torch.randn([3, 2, 1, 1]).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn([9, 8, 7, 6, 5, 4]),
    },
]

dtypes = [
    torch.float16,
    torch.float32,
    torch.uint8,
    torch.int8,
    # torch.int16,  # TODO(@mt-ai): unique test with int16 would throw UNKNOWN ERROR
    torch.int32,
    torch.int64,
]
if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", dtypes)
@pytest.mark.parametrize("sort", [True])
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [True, False])
def test_unique(input_data, data_type, sort, return_inverse, return_counts):
    unit_input = {}
    if isinstance(input_data["input"], torch.Tensor):
        unit_input["input"] = input_data["input"].to(data_type)
    unit_input["return_inverse"] = return_inverse
    unit_input["return_counts"] = return_counts
    unit_input["sorted"] = sort
    if data_type in [torch.float16, torch.bfloat16]:
        abs_diff, rel_diff = 5e-2, 5e-3
        test = testing.OpTest(
            func=torch.unique,
            input_args=unit_input,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        if data_type == torch.float16:
            test.check_musafp16_vs_musafp32()
        else:
            test.check_musabf16_vs_musafp16()
        test.check_grad_fn()
    else:
        abs_diff, rel_diff = 1e-5, 1e-6
        test = testing.OpTest(
            func=torch.unique,
            input_args=unit_input,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        test.check_result()
        test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", dtypes)
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [True, False])
def test_unique_consecutive(input_data, data_type, return_inverse, return_counts):
    unit_input = {}
    if isinstance(input_data["input"], torch.Tensor):
        unit_input["input"] = input_data["input"].to(data_type)
    # unit_input["return_inverse"] = return_inverse
    unit_input["return_counts"] = return_counts
    unit_input["return_inverse"] = return_inverse

    if data_type in [torch.float16, torch.bfloat16]:
        abs_diff, rel_diff = 5e-2, 5e-3
        test = testing.OpTest(
            func=torch.unique_consecutive,
            input_args=unit_input,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        if data_type == torch.float16:
            test.check_musafp16_vs_musafp32()
        else:
            test.check_musabf16_vs_musafp16()
        test.check_grad_fn()
    else:
        abs_diff, rel_diff = 1e-5, 1e-6
        test = testing.OpTest(
            func=torch.unique_consecutive,
            input_args=unit_input,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        test.check_result()
        test.check_grad_fn()


input_data = [
    {"input": torch.tensor([[1, 0, 1, 3], [1, 2, 1, 2], [1, 0, 1, 3]]), "dim": 0},
    {"input": torch.tensor([[1, 0, 1, 3], [1, 2, 1, 2], [1, 0, 1, 3]]), "dim": 1},
    {"input": torch.randint(0, 3, [16, 8, 4]), "dim": -1},
    {"input": torch.randint(0, 3, [20, 10, 4])[:, ::2], "dim": 1},
]
dtypes = [
    torch.float16,
    torch.float32,
    torch.uint8,
    torch.int8,
    # torch.int16,  # TODO(@mt-ai): unique test with int16 would throw UNKNOWN ERROR
    torch.int32,
    torch.int64,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", dtypes)
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [True, False])
def test_unique_dim(input_data, data_type, return_inverse, return_counts):
    unit_input = {}
    if isinstance(input_data["input"], torch.Tensor):
        unit_input["input"] = input_data["input"].to(data_type)
    unit_input["return_counts"] = return_counts
    unit_input["return_inverse"] = return_inverse
    unit_input["dim"] = input_data["dim"]

    if data_type in [torch.float16, torch.bfloat16]:
        abs_diff, rel_diff = 5e-2, 5e-3
        test = testing.OpTest(
            func=torch.unique,
            input_args=unit_input,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        if data_type == torch.float16:
            test.check_musafp16_vs_musafp32()
        else:
            test.check_musabf16_vs_musafp16()
        test.check_grad_fn()
    else:
        abs_diff, rel_diff = 1e-5, 1e-6
        test = testing.OpTest(
            func=torch.unique,
            input_args=unit_input,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        test.check_result()
        test.check_grad_fn()
