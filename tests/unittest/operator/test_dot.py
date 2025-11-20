"""Test dot operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    {
        "input": torch.randn(0),
        "tensor": torch.randn(0),
    },
    {
        "input": torch.randn(4),
        "tensor": torch.randn(4),
    },
    {
        "input": torch.randn(1024),
        "tensor": torch.randn(1024),
    },
    {
        "input": torch.randn(256),
        "tensor": torch.randn(256),
    },
    {
        "input": torch.randn(256)[1:20:2],
        "tensor": torch.randn(256)[1:20:2],
    },
]

complex_input_data = [
    {
        "input": torch.randn(0, dtype=torch.complex64),
        "tensor": torch.randn(0, dtype=torch.complex64),
    },
    {
        "input": torch.zeros(1, dtype=torch.complex64),
        "tensor": torch.zeros(1, dtype=torch.complex64),
    },
    {
        "input": torch.randn(4, dtype=torch.complex64),
        "tensor": torch.randn(4, dtype=torch.complex64),
    },
    {
        "input": torch.randn(256, dtype=torch.complex128),
        "tensor": torch.randn(256, dtype=torch.complex128),
    },
    {
        "input": torch.randn(256, dtype=torch.complex64)[1:20:2],
        "tensor": torch.randn(256, dtype=torch.complex64)[1:20:2],
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_dot(input_data):
    test = testing.OpTest(
        func=torch.dot,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_dot_fp16(input_data):
    test = testing.OpTest(
        func=torch.dot,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3),
    )
    test.check_musafp16_vs_musafp32()
    test.check_grad_fn(fp16=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data + complex_input_data)
def test_vdot(input_data):
    inp0_cpu = input_data["input"]
    inp1_cpu = input_data["tensor"]

    inp0_musa = inp0_cpu.musa()
    inp1_musa = inp1_cpu.musa()

    out_cpu = torch.vdot(inp0_cpu, inp1_cpu)
    out_musa = torch.vdot(inp0_musa, inp1_musa).cpu()

    assert torch.allclose(out_cpu, out_musa, rtol=1e-3, atol=1e-3), \
        "vdot fail"
