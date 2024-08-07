"""Test lerf operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    {
        "input": torch.randn(0),
        "end": torch.randn(0),
        "weight": torch.randn(0),
    },
    {
        "input": torch.randn(0),
        "end": torch.randn(0),
        "weight": 0.5,
    },
    {
        "input": torch.randn(10),
        "end": torch.randn(10),
        "weight": torch.randn(10),
    },
    {
        "input": torch.randn(10),
        "end": torch.randn(10),
        "weight": 2.7,
    },
    {
        "input": torch.randn(10),
        "end": torch.randn(10, 10),
        "weight": torch.randn(10),
    },
    {
        "input": torch.randn(10),
        "end": torch.randn(10, 10),
        "weight": 2.7,
    },
    {
        "input": torch.randn(8, 16),
        "end": torch.randn(8, 1),
        "weight": torch.randn(8, 1),
    },
    {
        "input": torch.randn(8, 16),
        "end": torch.randn(8, 1),
        "weight": 2.7,
    },
]

dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.float64]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_lerp(input_data, dtype):
    if dtype in [torch.float16, torch.bfloat16]:
        if testing.get_musa_arch() < 22:
            return
        abs_diff, rel_diff = 5e-2, 5e-3
        test = testing.OpTest(
            func=torch.lerp,
            input_args=input_data,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        test.check_musafp16_vs_musafp32()
        test.check_musabf16_vs_musafp16()
    else:
        abs_diff, rel_diff = 1e-3, 1e-3
        test = testing.OpTest(
            func=torch.lerp,
            input_args=input_data,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_lerp_out(input_data, dtype):
    max_ele = 0
    for k, v in input_data.items():
        if isinstance(v, torch.Tensor):
            # input_data[k] = input_data[k].to(dtype)
            if input_data[k].numel() >= max_ele:
                out = torch.empty_like(input_data[k])
                max_ele = out.numel()
    input_data["out"] = out

    if dtype in [torch.float16, torch.bfloat16]:
        if testing.get_musa_arch() < 22:
            return
        abs_diff, rel_diff = 5e-2, 5e-3
        test = testing.OpTest(
            func=torch.lerp,
            input_args=input_data,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        test.check_musafp16_vs_musafp32()
        test.check_musabf16_vs_musafp16()
    else:
        abs_diff, rel_diff = 1e-3, 1e-3
        test = testing.OpTest(
            func=torch.lerp,
            input_args=input_data,
            comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
        )
        test.check_result()
