"""Test reflection_pad operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
from torch import nn
import pytest

from torch_musa import testing


input_data = [
    torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3),
    torch.arange(200, dtype=torch.float).reshape(2, 10, 10),
    torch.rand(0, 4, 4, 4),
    torch.rand(0, 4, 4, 4).to(memory_format=torch.channels_last),
]

# not support for fp16 and int
support_dtypes = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_reflection_pad2d(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReflectionPad2d((1, 1, 2, 0))
    test = testing.OpTest(func=m, input_args={"input": input_data})
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        torch.randn([1, 10], requires_grad=True),
        torch.randn([20, 30], requires_grad=True),
        torch.randn([1, 20, 40], requires_grad=True),
        torch.randn([1, 10, 30], requires_grad=True),
    ],
)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("pad", [(3, 1), (2, 2)])
@pytest.mark.parametrize("mode", ["reflect"])
def test_reflection_pad1d(input_data, dtype, pad, mode):
    input_data = input_data.to(dtype)
    test = testing.OpTest(
        func=torch.nn.functional.pad,
        input_args={"input": input_data, "pad": pad, "mode": mode},
    )
    test.check_result(train=True)


input_data = [
    torch.arange(8, dtype=torch.float).reshape(1, 2, 4),
    torch.arange(200, dtype=torch.float).reshape(2, 10, 10),
    torch.rand(0, 10, 10),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_replication_pad1d(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReplicationPad1d(2)
    output_cpu = m(input_data)
    output_musa = m(input_data.to("musa"))
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")


input_data = [
    torch.arange(32, dtype=torch.float).reshape(1, 2, 4, 4),
    torch.arange(4000, dtype=torch.float).reshape(2, 10, 10, 20),
    torch.rand(0, 10, 10, 8),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_replication_pad2d(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReplicationPad2d(3)
    output_cpu = m(input_data)
    output_musa = m(input_data.to("musa"))
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")


input_data = [
    torch.arange(128, dtype=torch.float).reshape(1, 2, 4, 4, 4),
    torch.arange(32000, dtype=torch.float).reshape(2, 10, 10, 20, 8),
    torch.rand(0, 10, 10, 8, 16),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_replication_pad3d(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReplicationPad3d(4)
    output_cpu = m(input_data)
    output_musa = m(input_data.to("musa"))
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")
