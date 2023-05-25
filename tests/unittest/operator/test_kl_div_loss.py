"""Test kl_div operator."""
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

reduction = ["batchmean", "sum"]

all_support_types = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("reduction", reduction)
def test_kl_div(input_data, dtype, reduction):
    func = partial(F.kl_div, reduction=reduction)
    reduction_dim = input_data["input"].dim() - 1
    cpu_input = input_data["input"].to(dtype)
    musa_input = input_data["input"].to(dtype)

    cpu_target = input_data["target"]
    musa_target = input_data["target"].to("musa")

    cpu_output = func(F.log_softmax(cpu_input, dim=reduction_dim),
                      F.softmax(cpu_target, dim=reduction_dim))
    musa_output = func(F.log_softmax(musa_input.to("musa"), dim=reduction_dim),
                       F.softmax(musa_target.to("musa"), dim=reduction_dim))

    comparator = testing.DefaultComparator()
    assert comparator(cpu_output, musa_output.cpu())

    cpu_output.sum().backward()
    musa_output.sum().backward()
    # import pdb; pdb.set_trace();
    assert comparator(cpu_input.grad, musa_input.grad.cpu())
