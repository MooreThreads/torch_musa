"""Test bmm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(4, 10, 5),
        "mat2": torch.randn(4, 5, 10),
    },
    {
        "input": torch.randn(4, 10, 0),
        "mat2": torch.randn(4, 0, 10),
    },
    {
        "input": torch.randn(4, 4).T.unsqueeze(0),
        "mat2": torch.randn(4, 4).T.unsqueeze(0),
    },
    {
        "input": torch.randn(6, 4).T.unsqueeze(0),
        "mat2": torch.randn(4, 6).T.unsqueeze(0),
    },
    {
        "input": torch.randn(7, 9).T.unsqueeze(0),
        "mat2": torch.randn(9, 7).T.unsqueeze(0),
    },
    {
        "input": torch.randn(4, 10, 5).transpose(1, 2),
        "mat2": torch.randn(4, 5, 10).transpose(1, 2),
    },
    {
        "input": torch.randn(4, 10, 5).transpose(0, 1),
        "mat2": torch.randn(5, 10, 10).transpose(0, 1),
    },
    {
        "input": torch.as_strided(torch.randn(128), (2, 64, 1), (64, 1, 0)),
        "mat2": torch.as_strided(torch.randn(128), (2, 1, 64), (64, 0, 1)),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_bmm(input_data):
    test = testing.OpTest(
        func=torch.bmm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


input_data_complex = [
    {
        "input": torch.randn(4, 10, 5, dtype=torch.complex64),
        "mat2": torch.randn(4, 5, 10, dtype=torch.complex64),
    },
    {
        "input": torch.randn(4, 10, 0, dtype=torch.complex128),
        "mat2": torch.randn(4, 0, 10, dtype=torch.complex128),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data_complex)
def test_bmm_complex(input_data):
    test = testing.OpTest(
        func=torch.bmm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


input_data_complex_mh = [
    {
        "input": torch.randn(4, 5, 5, dtype=torch.complex64).mH,
        "mat2": torch.randn(4, 5, 10, dtype=torch.complex64),
    },
    {
        "input": torch.randn(4, 5, 10, dtype=torch.complex64),
        "mat2": torch.randn(4, 5, 10, dtype=torch.complex64).mH,
    },
    {
        "input": torch.randn(4, 5, 5, dtype=torch.complex64).mH,
        "mat2": torch.randn(4, 5, 5, dtype=torch.complex64).mH,
    },
    {
        "input": torch.randn(4, 5, 5, dtype=torch.complex128).mH,
        "mat2": torch.randn(4, 5, 10, dtype=torch.complex128),
    },
]


@pytest.mark.parametrize("input_data", input_data_complex_mh)
def test_bmm_complex_mh(input_data):
    input_cpu, mat2_cpu = input_data["input"], input_data["mat2"]
    input_musa = input_cpu.musa()
    mat2_musa = mat2_cpu.musa()

    out_musa = torch.bmm(input_musa, mat2_musa)
    out_cpu = torch.bmm(input_cpu, mat2_cpu)

    torch.allclose(out_musa.cpu(), out_cpu)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1),
        (1, 1, 256, 256),
        (4, 256, 256, 1),
        (4, 16, 64, 32),
        (4, 64, 128, 256),
        (4, 128, 256, 512),
        (4, 7168, 4480, 7168),
    ],
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_bmm_out_dtype_fp32(shape):
    batch, m, k, n = shape

    a_bf16 = torch.randn(
        batch,
        m,
        k,
        device="musa",
        dtype=torch.bfloat16,
    )
    b_bf16 = torch.randn(
        batch,
        n,
        k,
        device="musa",
        dtype=torch.bfloat16,
    ).transpose(1, 2)

    a_fp16 = a_bf16.to(torch.float16)
    b_fp16 = b_bf16.to(torch.float16)

    # bf16 input -> fp32 output
    bf16_result = torch.bmm(
        a_bf16,
        b_bf16,
        out_dtype=torch.float32,
    )

    # fp16 input -> fp32 output
    fp16_result = torch.bmm(
        a_fp16,
        b_fp16,
        out_dtype=torch.float32,
    )

    # fp32 input -> fp32 output, used as reference
    fp32_result = torch.bmm(
        a_bf16.to(torch.float32),
        b_bf16.to(torch.float32),
    )

    assert bf16_result.dtype == torch.float32
    assert fp16_result.dtype == torch.float32
    assert fp32_result.dtype == torch.float32

    torch.testing.assert_close(
        bf16_result,
        fp32_result,
        rtol=1e-3,
        atol=1e-3,
    )

    torch.testing.assert_close(
        fp16_result,
        fp32_result,
        rtol=1e-3,
        atol=1e-3,
    )
