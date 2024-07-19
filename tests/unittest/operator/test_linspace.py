"""Test linspace operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random
from functools import partial
import pytest
import torch
import torch.nn.functional as F
from torch_musa import testing

input_datas = [
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)},
]

all_support_types = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
def test_func(input_data, dtype):
    begin_end = input_data["input"]
    num = input_data["num"]

    cpu_input = torch.linspace(begin_end[0], begin_end[1], num, device="cpu").to(dtype)
    musa_input = torch.linspace(begin_end[0], begin_end[1], num, device="musa").to(
        dtype
    )

    comparator = testing.DefaultComparator()
    assert comparator(cpu_input, musa_input.cpu())

    out_res = torch.empty_like(musa_input)
    prev_addr = out_res.data_ptr()
    out_output = torch.linspace(begin_end[0], begin_end[1], num, out=out_res)

    assert prev_addr == out_res.data_ptr()
    assert out_output.data_ptr() == out_res.data_ptr()
    assert comparator(out_res.cpu(), musa_input.cpu())
    assert comparator(out_output.cpu(), out_res.cpu())
