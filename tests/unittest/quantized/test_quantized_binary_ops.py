"""Test quantized binary operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import torch.ao.nn.quantized as nnq
import pytest
import torch_musa

from torch_musa import testing

torch.manual_seed(41)

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
    {
        "input_a": torch.randn(2, 32, 32, 16),
        "input_b": torch.randn(2, 32, 32, 16),
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
    if dtype == torch.qint8:
        zero_point = 0
    else:
        zero_point = 128
    qin_a = torch.quantize_per_tensor(
        input_a, input_a.abs().max() / 2**7, zero_point, dtype
    )
    qin_b = torch.quantize_per_tensor(
        input_b, input_a.abs().max() / 2**7, zero_point, dtype
    )

    if qin_a.dim == 4:
        qin_a = qin_a.permute(0, 3, 1, 2)
        qin_b = qin_b.permute(0, 3, 1, 2)

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
