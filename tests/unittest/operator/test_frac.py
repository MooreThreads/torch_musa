"""Test frac operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

# Note: muDNN doesn't support float64 or bool for this operator.
# We should enable these two types after fill is implemented with MUSA.
data_type = testing.get_all_support_types()

input_data = [
    {"input": torch.tensor(5.0)},
    {"input": torch.tensor([5.1, 3.22, 2.333])},
    {"input": torch.tensor([1, 2.5, -3.2])},
    {"input": torch.tensor([[5, 3, 1, 2, 3]])},
    {"input": torch.randn(1)},
    {"input": torch.randn(1, 2)},
    {"input": torch.randn(1, 2, 3)},
    {"input": torch.randn(1, 2, 3)},
    {"input": torch.randn(1, 0, 3)},
    {"input": torch.randn(1, 2, 3, 4)},
    {"input": torch.randn(1, 2, 3, 4, 3)},
    {"input": torch.randn(1, 2, 3, 4, 3, 2)},
    {"input": torch.randn(1, 2, 3, 4, 3, 2, 4)},
    {"input": torch.randn(1, 2, 3, 4, 3, 2, 4, 2)},
]


for data in testing.get_raw_data():
    input_data.append({"input": data})


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_frac(input_data, dtype):
    test = testing.OpTest(
        func=torch.frac,
        input_args={
            "input": input_data["input"].to(dtype)
        },
    )
    test.check_result()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_frac_(input_data, dtype):
    cpu_result = input_data["input"].to(dtype)
    cpu_result.frac()

    musa_result = input_data["input"].to(dtype).to("musa")
    musa_result.frac()

    assert testing.DefaultComparator()(cpu_result, musa_result.cpu())
    assert cpu_result.shape == musa_result.shape
    assert cpu_result.dtype == musa_result.dtype

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_frac_out(input_data, dtype):
    cpu_input = input_data["input"].to(dtype)
    cpu_result = torch.zeros_like(input_data["input"]).to(dtype)
    torch.frac(cpu_input, out=cpu_result)

    musa_input = input_data["input"].to(dtype).to("musa")
    musa_result = torch.zeros_like(input_data["input"]).to(dtype).to("musa")
    torch.frac(musa_input, out=musa_result)

    assert testing.DefaultComparator()(cpu_result, musa_result.cpu())
    assert cpu_result.shape == musa_result.shape
    assert cpu_result.dtype == musa_result.dtype
