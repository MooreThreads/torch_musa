"""Test logspace operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random
from functools import partial
import pytest
import torch
import torch.nn.functional as F
from torch_musa import testing

input_datas = [
    {"input": [random.random(), random.random()], "num": random.randint(0, 100000000)}
    for _ in range(12)
]

all_support_types = [torch.float32]
bases = [10.0, 2.0]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("base", bases)
def test_logspace_func(input_data, dtype, base):
    begin_end = input_data["input"]
    num = input_data["num"]

    cpu_input = torch.logspace(
        begin_end[0], begin_end[1], num, base=base, device="cpu"
    ).to(dtype)
    musa_input = torch.logspace(
        begin_end[0], begin_end[1], num, base=base, device="musa"
    ).to(dtype)

    comparator = testing.DefaultComparator()
    assert comparator(cpu_input, musa_input.cpu())

    out_res = torch.empty_like(musa_input)
    prev_addr = out_res.data_ptr()
    out_output = torch.logspace(begin_end[0], begin_end[1], num, base=base, out=out_res)

    assert prev_addr == out_res.data_ptr()
    assert out_output.data_ptr() == out_res.data_ptr()
    assert comparator(out_res.cpu(), musa_input.cpu())
    assert comparator(out_output.cpu(), out_res.cpu())
