"""Test quantized activations."""

# pylint: disable=missing-function-docstring, unused-import
import pytest
import torch
import torch_musa

from torch_musa import testing

torch.manual_seed(41)


def function(input_data, func, dtype=None):
    if dtype is not None and isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(
        func=func, input_args=input_data, comparators=testing.QuantizedComparator()
    )
    test.check_result()


input_activation = [
    {"input": torch.quantize_per_tensor(torch.randn(3, 4), 0.03, 127, torch.quint8)},
    {"input": torch.quantize_per_tensor(torch.randn(3, 4, 6), 0.02, 0, torch.qint8)},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_activation)
def test_relu(input_data):
    function(input_data, torch.relu)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_activation)
def test_relu_(input_data):
    function(input_data, torch.relu_)


# cpu quantized sigmoid generates wrong values, disable UT for now
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_activation)
def notest_sigmoid(input_data):
    function(input_data, torch.nn.functional.sigmoid)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_activation)
def test_gelu(input_data):
    function(input_data, torch.nn.functional.gelu)
