"""Test kl_div operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
import torch.nn.functional as F
from torch_musa import testing


input_datas = [
    {"input": torch.randn([1, 10]), "target": torch.randn([1, 10])},
    {"input": torch.randn([2, 3, 6]), "target": torch.randn([2, 3, 6])},
    {"input": torch.randn([0, 0, 0]), "target": torch.randn([0, 0, 0])},
    {"input": torch.randn([2, 3, 6, 10]), "target": torch.randn([2, 3, 6, 10])},
    {
        "input": torch.randn([2, 3, 6, 10]).to(memory_format=torch.channels_last),
        "target": torch.randn([2, 3, 6, 10]).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn([0, 3, 0, 10]).to(memory_format=torch.channels_last),
        "target": torch.randn([0, 3, 0, 10]).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn([2, 3, 1, 1]).to(memory_format=torch.channels_last),
        "target": torch.randn([2, 3, 1, 1]).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn([2, 1, 7, 4]).to(memory_format=torch.channels_last),
        "target": torch.randn([2, 1, 7, 4]).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn([2, 0, 7, 4]).to(memory_format=torch.channels_last),
        "target": torch.randn([2, 0, 7, 4]).to(memory_format=torch.channels_last),
    },
    {"input": torch.randn([2, 3, 6, 10, 20]), "target": torch.randn([2, 3, 6, 10, 20])},
    {"input": torch.randn([2, 3, 6, 0, 20]), "target": torch.randn([2, 3, 6, 0, 20])},
    {
        "input": torch.randn([4, 5, 6, 7, 8, 9]),
        "target": torch.randn([4, 5, 6, 7, 8, 9]),
    },
    {
        "input": torch.randn([4, 5, 6, 7, 8, 9, 16]),
        "target": torch.randn([4, 5, 6, 7, 8, 9, 16]),
    },
    {
        "input": torch.randn([4, 5, 6, 7, 8, 9, 16, 2]),
        "target": torch.randn([4, 5, 6, 7, 8, 9, 16, 2]),
    },
]

reduction = ["batchmean", "sum"]

all_support_types = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("reduction", reduction)
def test_kl_div(input_data, dtype, reduction):
    reduction_dim = input_data["input"].dim() - 1
    input_data["input"] = input_data["input"].to(dtype)

    input_tensor = F.log_softmax(input_data["input"], dim=reduction_dim)
    input_tensor.requires_grad_(True)
    target_tensor = F.softmax(input_data["target"], dim=reduction_dim)

    test = testing.OpTest(
        func=torch.nn.KLDivLoss,
        input_args={"reduction": reduction},
        comparators=testing.DefaultComparator(equal_nan=True),
    )
    test.check_result({"input": input_tensor, "target": target_tensor}, train=True)

    class_musa = torch.nn.KLDivLoss(reduction=reduction)(
        input_tensor.clone().musa().requires_grad_(),
        target_tensor.clone().musa().requires_grad_(),
    ).grad_fn.__class__

    class_cpu = torch.nn.KLDivLoss(reduction=reduction)(
        input_tensor.clone().cpu().requires_grad_(),
        target_tensor.clone().cpu().requires_grad_(),
    ).grad_fn.__class__

    assert class_cpu == class_musa
