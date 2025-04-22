"""Test mm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(4, 0),
        "mat2": torch.randn(0, 2),
    },
    {
        "input": torch.randn(0, 30),
        "mat2": torch.randn(30, 2),
    },
    {
        "input": torch.randn(2, 30),
        "mat2": torch.randn(30, 2),
    },
    {
        "input": torch.randn(30, 5).t(),
        "mat2": torch.randn(30, 2),
    },
    {
        "input": torch.randn(30, 5).t(),
        "mat2": torch.randn(2, 30).t(),
    },
    {
        "input": torch.randn(5, 30),
        "mat2": torch.randn(2, 30).t(),
    },
    {
        "input": torch.randn(64, 128),
        "mat2": torch.randn(256, 128).t(),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_mm(input_data):
    test = testing.OpTest(
        func=torch.mm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_mm_fp16(input_data):
    test = testing.OpTest(
        func=torch.mm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3),
    )
    test.check_musafp16_vs_musafp32()
    test.check_out_ops(fp16=True)
    test.check_grad_fn(fp16=True)


input_data = [
    {
        "mat1": torch.randn(16, 64),
        "mat2": torch.randn(64, 32),
    },
    {
        "mat1": torch.randn(32, 32).t(),
        "mat2": torch.randn(32, 32),
    },
    {
        "mat1": torch.randn(128, 64).t(),
        "mat2": torch.randn(128, 128).t(),
    },
    {
        "mat1": torch.randn(64, 128),
        "mat2": torch.randn(256, 128).t(),
    },
]


f8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]
E5M2MAX = 57344
E4M3MAX = 448


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", f8_dtypes)
@pytest.mark.parametrize(
    "out_dtype", f8_dtypes + [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.skipif(
    testing.get_musa_arch() < 31, reason="_scaled_mm only works on MUSA_ARCH>=31"
)
def test_scaled_mm(input_data, dtype, out_dtype):
    """
    c = a @ b -> c8 = a8 @ b8
    c8 = f8(c * scale_out) = f8(a * scale_a) @ f8(b * scale_b) = f8[scale_a * scale_b * (a @ b)]
    """
    if out_dtype in f8_dtypes and out_dtype != dtype:
        return
    if dtype == torch.float8_e4m3fn:
        fp8max = E4M3MAX
    else:
        fp8max = E5M2MAX
    mat1 = input_data["mat1"].clone()
    mat2 = input_data["mat2"].clone()
    # get f8 tensor from inputs
    scale_a = mat1.abs().max() / fp8max
    scale_b = mat2.abs().max() / fp8max
    f8_a = (mat1 / scale_a).to(dtype)
    f8_b = (mat2 / scale_b).to(dtype)

    # fp32 golden result
    golden = torch.mm(f8_a.float(), f8_b.float()) * scale_a * scale_b

    # out_dtype scaled_mm result
    scale_out = golden.abs().max() / fp8max
    musa_out, amax = torch._scaled_mm(
        f8_a.musa(),
        f8_b.musa(),
        scale_a=scale_a,
        scale_b=scale_b,
        scale_result=scale_out if out_dtype in f8_dtypes else None,
        out_dtype=out_dtype,
    )
    musa_out = musa_out.cpu().float()
    assert torch.allclose(musa_out.abs().max(), amax)

    if out_dtype in f8_dtypes:
        musa_out = musa_out * scale_out
    assert torch.allclose(
        golden, musa_out, rtol=0.25 if dtype == torch.float8_e5m2 else 0.125, atol=1e-2
    )
