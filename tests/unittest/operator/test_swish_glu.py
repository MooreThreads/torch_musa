"""Test swish_glu operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,invalid-name,not-callable, redefined-builtin
import torch
import torch.nn.functional as F
import pytest
import torch_musa

from torch_musa import testing


def swish_glu_cpu(input: torch.Tensor):
    in_tensor_h = input.to(torch.float32)
    inputs = torch.chunk(in_tensor_h, 2, dim=-1)
    ret = torch.nn.functional.silu(inputs[0]) * inputs[1]
    return ret.to(input.dtype)


input_datas = [
    torch.randn(1, 0, 12),
    torch.randn(16, 32),
    torch.randn(8, 128, 64),
    torch.randn(32, 1024, 128),
    torch.randn(64, 1024, 2048),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(testing.get_musa_arch() < 22, reason="only test on arch>=22")
@pytest.mark.parametrize("input", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_swiglu(input, dtype):
    input_data = {"input": input.clone().to(dtype)}
    atol = 1e-9
    if dtype == torch.half:
        atol = 1e-2
    elif dtype == torch.bfloat16:
        atol = 1e-3

    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-2
    elif dtype == torch.float32:
        atol, rtol = 1e-6, 1e-6
    else:
        atol, rtol = 5e-3, 5e-2
    test = testing.OpTest(
        func=F.swish_glu,
        refer_func=swish_glu_cpu,
        input_args=input_data,
        comparators=testing.DefaultComparator(rel_diff=rtol, abs_diff=atol),
    )
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
    else:
        test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(testing.get_musa_arch() < 22, reason="only test on arch>=22")
@pytest.mark.parametrize("input", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_swiglu_backward(input, dtype):
    input_data = {"input": torch.rand_like(input, dtype=dtype, requires_grad=True)}
    atol = 1e-9
    if dtype == torch.half:
        atol = 1e-2
    elif dtype == torch.bfloat16:
        atol = 1e-3

    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-2
    elif dtype == torch.float32:
        atol, rtol = 1e-6, 1e-6
    else:
        atol, rtol = 5e-3, 5e-2
    test = testing.OpTest(
        func=F.swish_glu,
        refer_func=swish_glu_cpu,
        input_args=input_data,
        comparators=testing.DefaultComparator(rel_diff=rtol, abs_diff=atol),
    )
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32(train=True)
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16(train=True)
    else:
        test.check_result(train=True)
