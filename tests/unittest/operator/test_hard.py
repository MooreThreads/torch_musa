"""Test hardswish, hardsigmoid."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
import torch.nn.functional as F
from torch_musa import testing


input_datas = [
    {"input": torch.randn([1, 10], requires_grad=True), "target": torch.randn([1, 10])},
    {"input": torch.randn([0, 0], requires_grad=True), "target": torch.randn([0, 0])},
    {
        "input": torch.randn([2, 3, 6], requires_grad=True),
        "target": torch.randn([2, 3, 6]),
    },
    {
        "input": torch.randn([2, 0, 6], requires_grad=True),
        "target": torch.randn([2, 0, 6]),
    },
    {
        "input": torch.randn([2, 3, 6, 10], requires_grad=True),
        "target": torch.randn([2, 3, 6, 10]),
    },
    {
        "input": torch.randn([2, 3, 6, 10], requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        "target": torch.randn([2, 3, 6, 10]).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn([2, 3, 1, 1], requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        "target": torch.randn([2, 3, 1, 1]).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn([2, 1, 6, 10], requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        "target": torch.randn([2, 1, 6, 10]).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn([2, 1, 6, 10], requires_grad=True).to(
            memory_format=torch.channels_last
        ),
        "target": torch.randn([2, 1, 6, 10]),
    },
    {
        "input": torch.randn([2, 3, 6, 10, 20], requires_grad=True),
        "target": torch.randn([2, 3, 6, 10, 20]),
    },
    {
        "input": torch.randn([4, 5, 6, 7, 8, 9], requires_grad=True),
        "target": torch.randn([4, 5, 6, 7, 8, 9]),
    },
    {
        "input": torch.randn([4, 5, 6, 7, 8, 9, 16], requires_grad=True),
        "target": torch.randn([4, 5, 6, 7, 8, 9, 16]),
    },
    {
        "input": torch.randn([4, 5, 6, 7, 8, 9, 16, 2], requires_grad=True),
        "target": torch.randn([4, 5, 6, 7, 8, 9, 16, 2]),
    },
    {
        "input": torch.randn([4, 0, 6, 7, 8, 9, 16, 2], requires_grad=True),
        "target": torch.randn([4, 0, 6, 7, 8, 9, 16, 2]),
    },
    {
        "input": torch.randn([4, 0, 6, 7, 0, 9, 0, 2], requires_grad=True),
        "target": torch.randn([4, 0, 6, 7, 0, 9, 0, 2]),
    },
]

all_support_types = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
def test_hardswish(input_data, dtype):
    input_data["input"] = input_data["input"].to(dtype)

    test = testing.OpTest(
        func=F.hardswish,
        input_args={"input": input_data["input"]},
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result(train=True)
    assert (
        F.hardswish(input_data["input"].clone().musa()).grad_fn.__class__
        == F.hardswish(input_data["input"].clone().cpu()).grad_fn.__class__
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
def test_hardsigmoid(input_data, dtype):
    input_data["input"] = input_data["input"].to(dtype)

    test = testing.OpTest(
        func=F.hardsigmoid,
        input_args={"input": input_data["input"]},
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result(train=True)
    assert str(
        F.hardsigmoid(input_data["input"].clone().musa()).grad_fn.__class__
    ) == str(F.hardsigmoid(input_data["input"].clone().cpu()).grad_fn.__class__)
