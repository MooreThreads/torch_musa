"""Test quantized binary operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.quantized as nniq
import pytest
import torch_musa

from torch_musa import testing


input_datas = [
    {
        "input_a": torch.randn(128, 512),
        "input_b": torch.randn(128, 512),
        "relu": True,
    },
    {
        "input_a": torch.randn(256, 256),
        "input_b": torch.randn(256, 256),
        "relu": False,
    },
]
dtypes = [torch.quint8, torch.qint8]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_qfunctional(input_data, dtype):
    """Test quantized functional operators"""
    input_a = input_data["input_a"]
    input_b = input_data["input_b"]
    qin_a = torch.quantize_per_tensor(input_a, 0.05, 10, dtype)
    qin_b = torch.quantize_per_tensor(input_b, 0.06, 5, dtype)

    input_args = {
        "x": qin_a,
        "y": qin_b,
    }
    module = nnq.QFunctional()
    if input_data["relu"]:
        test = testing.OpTest(
            func=module.add_relu,
            input_args=input_args,
            comparators=testing.QuantizedComparator(abs_diff=1),
        )
    else:
        test = testing.OpTest(
            func=module.add,
            input_args=input_args,
            comparators=testing.QuantizedComparator(abs_diff=1),
        )
    test.check_result()
