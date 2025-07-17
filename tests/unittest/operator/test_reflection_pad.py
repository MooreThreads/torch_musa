"""Test reflection_pad operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
from torch import nn
import pytest

from torch_musa import testing


# not support for fp16 and int
support_dtypes = [torch.float32]


input_data = [
    torch.empty(3, 4).uniform_(-2, 2).requires_grad_(),
    torch.empty(0, 3, 4).requires_grad_(),
    torch.empty(2, 3, 4).uniform_(-2, 2).requires_grad_(),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_reflection_pad1d(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReflectionPad1d((2, 0))
    test = testing.OpTest(func=m, input_args={"input": input_data})
    test.check_result(train=True)


input_data = [
    torch.empty(3, 4, 5).uniform_(-2, 2).requires_grad_(True),
    torch.empty(0, 3, 4, 5).requires_grad_(True),
    torch.empty(2, 3, 4, 5).uniform_(-2, 2).requires_grad_(True),
    torch.empty(2, 3, 4, 5)
    .uniform_(-2, 2)
    .to(memory_format=torch.channels_last)
    .requires_grad_(True),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_reflection_pad2d(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReflectionPad2d((1, 1, 2, 0))
    test = testing.OpTest(func=m, input_args={"input": input_data})
    test.check_result(train=True)


input_data = [
    torch.empty(3, 4, 5, 6).uniform_(-2, 2).requires_grad_(True),
    torch.empty(0, 3, 4, 5, 6).requires_grad_(True),
    torch.empty(2, 3, 4, 5, 6).uniform_(-2, 2).requires_grad_(True),
    torch.empty(2, 3, 4, 5, 6)
    .uniform_(-2, 2)
    .to(memory_format=torch.channels_last_3d)
    .requires_grad_(True),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_reflection_pad3d(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReflectionPad3d((1, 1, 1, 1, 1, 1))
    test = testing.OpTest(func=m, input_args={"input": input_data})
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


input_2d_data_grad = [
    torch.rand(2, 3, 4, 4, requires_grad=True),
    torch.rand(0, 3, 4, 4, requires_grad=True),
    torch.rand(2, 3, 5, 6, requires_grad=True),
    torch.rand(2, 3, 5, 6).to(memory_format=torch.channels_last).requires_grad_(),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_2d_data_grad)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_replication_pad2d_backward(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReplicationPad2d((1, 1, 1, 1))
    test = testing.OpTest(func=m, input_args={"input": input_data})
    test.check_result(train=True)


input_3d_data_grad = [
    torch.rand(2, 3, 4, 4, 4, requires_grad=True),
    torch.rand(0, 3, 4, 4, 4, requires_grad=True),
    torch.rand(2, 3, 5, 6, 7, requires_grad=True),
    torch.rand(2, 3, 5, 6, 7).to(memory_format=torch.channels_last_3d).requires_grad_(),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_3d_data_grad)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_replication_pad3d_backward(input_data, dtype):
    input_data = input_data.to(dtype)
    m = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
    test = testing.OpTest(func=m, input_args={"input": input_data})
    test.check_result(train=True)
