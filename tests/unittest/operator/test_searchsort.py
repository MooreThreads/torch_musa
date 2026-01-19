"Test searchsort"
# pylint: disable=missing-function-docstring,missing-module-docstring,unused-import,redefined-builtin,redefined-outer-name

import torch
import pytest
from torch_musa import testing

searchsorted_inputs = [
    {
        "sorted_sequence": torch.tensor([1, 3, 5, 7, 9]),
        "values": torch.tensor([2, 4, 6, 8]),
        "right": False,
        "out_int32": False,
    },
    {
        "sorted_sequence": torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0]),
        "values": torch.tensor([2.0, 4.0, 6.0, 8.0]),
        "right": True,
        "out_int32": False,
    },
    {
        "sorted_sequence": torch.tensor([[1, 3, 5], [2, 4, 6]]),
        "values": torch.tensor([[2, 5], [3, 4]]),
        "right": False,
        "out_int32": False,
    },
    {
        "sorted_sequence": torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
        "values": torch.tensor([[2.0, 5.0], [3.0, 4.0]]),
        "right": True,
        "out_int32": False,
    },
    {
        "sorted_sequence": torch.tensor([10, 20, 30, 40, 50]),
        "values": torch.tensor([5, 15, 25, 35, 45, 55]),
        "right": False,
        "out_int32": False,
    },
    {
        "sorted_sequence": torch.tensor([1, 3, 5, 7, 9]),
        "values": torch.tensor([2, 4, 6, 8]),
        "right": False,
        "out_int32": True,
    },
]


scalar_inputs = [
    {
        "sorted_sequence": torch.tensor([1, 3, 5, 7, 9]),
        "values": 4.5,  # scalar
        "right": False,
        "out_int32": False,
    },
    {
        "sorted_sequence": torch.tensor([10, 20, 30, 40, 50]),
        "values": 25,  # scalar
        "right": False,
        "out_int32": True,
    },
]

dtypes = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.int16,
    torch.int8,
    torch.uint8,
]

if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", searchsorted_inputs)
@pytest.mark.parametrize("dtype", dtypes)
def test_searchsorted_tensor(inputs, dtype):

    if dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.int16]:
        if (
            not inputs["sorted_sequence"].is_floating_point()
            and dtype.is_floating_point
        ):
            return
        if (
            inputs["sorted_sequence"].is_floating_point()
            and not dtype.is_floating_point
        ):
            return

    sorted_sequence = inputs["sorted_sequence"].to(dtype)
    values = inputs["values"].to(dtype if dtype.is_floating_point else torch.float32)

    input_args = {
        "sorted_sequence": sorted_sequence,
        "input": values,
        "right": inputs["right"],
        "out_int32": inputs["out_int32"],
    }

    test = testing.OpTest(
        func=torch.searchsorted,
        input_args=input_args,
        comparators=testing.DefaultComparator(),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", searchsorted_inputs)
@pytest.mark.parametrize("dtype", dtypes)
def test_searchsorted_tensor_out(inputs, dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.int16]:
        if (
            not inputs["sorted_sequence"].is_floating_point()
            and dtype.is_floating_point
        ):
            return
        if (
            inputs["sorted_sequence"].is_floating_point()
            and not dtype.is_floating_point
        ):
            return

    sorted_sequence = inputs["sorted_sequence"].to(dtype)
    values = inputs["values"].to(dtype if dtype.is_floating_point else torch.float32)

    if inputs["out_int32"]:
        out_dtype = torch.int32
    else:
        out_dtype = torch.int64

    out = torch.empty(values.shape, dtype=out_dtype)

    input_args = {
        "sorted_sequence": sorted_sequence,
        "input": values,
        "right": inputs["right"],
        "out_int32": inputs["out_int32"],
        "out": out,
    }

    test = testing.OpTest(
        func=torch.searchsorted,
        input_args=input_args,
        comparators=testing.DefaultComparator(),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", scalar_inputs)
@pytest.mark.parametrize("dtype", dtypes)
def test_searchsorted_scalar(inputs, dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.int16]:
        if (
            not inputs["sorted_sequence"].is_floating_point()
            and dtype.is_floating_point
        ):
            return
        if (
            inputs["sorted_sequence"].is_floating_point()
            and not dtype.is_floating_point
        ):
            return

    sorted_sequence = inputs["sorted_sequence"].to(dtype)

    if isinstance(inputs["values"], (int, float)):
        values = inputs["values"]
    else:
        values = (
            inputs["values"].item()
            if inputs["values"].numel() == 1
            else inputs["values"]
        )

    input_args = {
        "sorted_sequence": sorted_sequence,
        "self": values,
        "right": inputs["right"],
        "out_int32": inputs["out_int32"],
    }

    test = testing.OpTest(
        func=torch.searchsorted,
        input_args=input_args,
        comparators=testing.DefaultComparator(),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", scalar_inputs)
@pytest.mark.parametrize("dtype", dtypes)
def test_searchsorted_scalar_out(inputs, dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.int16]:
        if (
            not inputs["sorted_sequence"].is_floating_point()
            and dtype.is_floating_point
        ):
            return
        if (
            inputs["sorted_sequence"].is_floating_point()
            and not dtype.is_floating_point
        ):
            return

    sorted_sequence = inputs["sorted_sequence"].to(dtype)

    if isinstance(inputs["values"], (int, float)):
        values = inputs["values"]
    else:
        values = (
            inputs["values"].item()
            if inputs["values"].numel() == 1
            else inputs["values"]
        )

    if inputs["out_int32"]:
        out_dtype = torch.int32
    else:
        out_dtype = torch.int64

    out = torch.empty([1], dtype=out_dtype)

    input_args = {
        "sorted_sequence": sorted_sequence,
        "self": values,
        "right": inputs["right"],
        "out_int32": inputs["out_int32"],
        "out": out,
    }

    test = testing.OpTest(
        func=torch.searchsorted,
        input_args=input_args,
        comparators=testing.DefaultComparator(),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", dtypes)
def test_searchsorted_edge_cases(dtype):
    if dtype in [torch.float16, torch.bfloat16]:
        return

    sorted_empty = torch.tensor([], dtype=dtype)
    values = torch.tensor(
        [1, 2, 3], dtype=dtype if dtype.is_floating_point else torch.float32
    )

    input_args = {
        "sorted_sequence": sorted_empty,
        "input": values,
        "right": False,
        "out_int32": True,
    }

    test = testing.OpTest(
        func=torch.searchsorted,
        input_args=input_args,
        comparators=testing.DefaultComparator(),
    )
    test.check_result()

    sorted_single = torch.tensor([5], dtype=dtype)
    values = torch.tensor(
        [1, 5, 10], dtype=dtype if dtype.is_floating_point else torch.float32
    )

    input_args = {
        "sorted_sequence": sorted_single,
        "input": values,
        "right": False,
        "out_int32": True,
    }

    test = testing.OpTest(
        func=torch.searchsorted,
        input_args=input_args,
        comparators=testing.DefaultComparator(),
    )
    test.check_result()

    sorted_same = torch.tensor([1, 1, 1, 2, 2, 3], dtype=dtype)
    values = torch.tensor(
        [0, 1, 2, 3, 4], dtype=dtype if dtype.is_floating_point else torch.float32
    )

    input_args = {
        "sorted_sequence": sorted_same,
        "input": values,
        "right": False,
        "out_int32": True,
    }

    test = testing.OpTest(
        func=torch.searchsorted,
        input_args=input_args,
        comparators=testing.DefaultComparator(),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", dtypes)
def test_searchsorted_sorter(dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.int16]:
        return

    unsorted = torch.tensor([5, 2, 8, 1, 9], dtype=dtype)
    values = torch.tensor(
        [3, 6, 1], dtype=dtype if dtype.is_floating_point else torch.float32
    )

    sorter = torch.argsort(unsorted)

    input_args = {
        "sorted_sequence": unsorted,
        "input": values,
        "right": False,
        "sorter": sorter,
        "out_int32": True,
    }

    test = testing.OpTest(
        func=torch.searchsorted,
        input_args=input_args,
        comparators=testing.DefaultComparator(),
    )
    test.check_result()
