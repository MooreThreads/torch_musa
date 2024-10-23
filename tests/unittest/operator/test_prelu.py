"""Test prelu operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
from torch_musa import testing


input_datas = [
    torch.randn([0, 0], requires_grad=True),
    torch.randn([1, 10], requires_grad=True),
    torch.randn([2, 3, 6], requires_grad=True),
    torch.randn([2, 3, 6, 10], requires_grad=True),
    torch.randn([2, 1, 6, 10], requires_grad=True).to(
        memory_format=torch.channels_last
    ),
    torch.randn([2, 3, 1, 1], requires_grad=True).to(memory_format=torch.channels_last),
    torch.randn([2, 3, 4, 5], requires_grad=True).to(memory_format=torch.channels_last),
    torch.randn([0, 3, 0, 5], requires_grad=True).to(memory_format=torch.channels_last),
    torch.randn([2, 3, 6, 10, 20], requires_grad=True),
    torch.randn([4, 5, 6, 7, 8, 9], requires_grad=True),
    torch.randn([4, 5, 6, 7, 8, 9, 16], requires_grad=True),
    torch.randn([4, 5, 6, 7, 8, 9, 16, 2], requires_grad=True),
    torch.randn([4, 5, 0, 7, 8, 0, 16, 2], requires_grad=True),
]

# only support fp32, even on cpu native
all_support_types = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("init", [0.15, 0.25])
@pytest.mark.parametrize("dtype", all_support_types)
def test_act(input_data, init, dtype):
    test = testing.OpTest(
        func=torch.nn.PReLU, input_args={"init": init, "dtype": dtype}, test_dtype=dtype
    )
    test.check_result(inputs={"input": input_data.to(dtype)}, train=True)
