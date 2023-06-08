"""Test prelu operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from functools import partial
import pytest
import torch
import torch.nn.functional as F
from torch_musa import testing


input_datas = [
    {"input": torch.randn([1, 10], requires_grad=True),
     "target": torch.randn([1, 10])},
    {"input": torch.randn([2, 3, 6], requires_grad=True),
     "target": torch.randn([2, 3, 6])},
     {"input": torch.randn([2, 3, 6, 10], requires_grad=True),
     "target": torch.randn([2, 3, 6, 10])},
     {"input": torch.randn([2, 3, 6, 10, 20], requires_grad=True),
     "target": torch.randn([2, 3, 6, 10, 20])},
     {"input": torch.randn([4, 5, 6, 7, 8, 9], requires_grad=True),
     "target": torch.randn([4, 5, 6, 7, 8, 9])},
     {"input": torch.randn([4, 5, 6, 7, 8, 9, 16], requires_grad=True),
     "target": torch.randn([4, 5, 6, 7, 8, 9, 16])},
     {"input": torch.randn([4, 5, 6, 7, 8, 9, 16, 2], requires_grad=True),
     "target": torch.randn([4, 5, 6, 7, 8, 9, 16, 2])},
]

# only support fp32, even on cpu native
all_support_types = [torch.float32]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
def test_act(input_data, dtype):
    act = torch.nn.PReLU()
    act_musa = torch.nn.PReLU()
    act_musa.to("musa")

    cpu_input = input_data["input"].to(dtype)
    musa_input = input_data["input"].to(dtype)

    cpu_output = act(cpu_input)
    musa_output = act_musa(musa_input.to("musa"))

    comparator = testing.DefaultComparator()
    assert comparator(cpu_output, musa_output.cpu())

    cpu_output.sum().backward()
    musa_output.sum().backward()
    #TODO: (mingyuan.wang) here musa_input and cpu_input refer to the same object actually.
    assert comparator(cpu_input.grad, musa_input.grad.cpu())
