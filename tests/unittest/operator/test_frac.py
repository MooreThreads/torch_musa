"""Test frac operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import copy
import torch
import pytest
import torch_musa
from torch_musa import testing

# Note: muDNN doesn't support float64 or bool for this operator.
# We should enable these two types after fill is implemented with MUSA.
data_type = testing.get_all_support_types()
float_dtypes = [torch.float32, torch.float16]
# bf16 is not supported on arch older than qy2
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)

input_data = [
    {"input": torch.tensor(5.0)},
    {"input": torch.tensor([5.1, 3.22, 2.333])},
    {"input": torch.tensor([1, 2.5, -3.2])},
    {"input": torch.tensor([[5, 3, 1, 2, 3]])},
    {"input": torch.randn(1)},
    {"input": torch.randn(1, 2)},
    {"input": torch.randn(1, 2, 3)},
    {"input": torch.randn(0, 0, 0)},
    {"input": torch.randn(1, 2, 3)},
    {"input": torch.randn(1, 0, 3)},
    {"input": torch.randn(1, 2, 3, 4)},
    {"input": torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)},
    {"input": torch.randn(4, 1, 3, 4).to(memory_format=torch.channels_last)},
    {"input": torch.randn(4, 2, 1, 1).to(memory_format=torch.channels_last)},
    {"input": torch.randn(4, 2, 8, 3).to(memory_format=torch.channels_last)},
    {"input": torch.randn(4, 2, 0, 0).to(memory_format=torch.channels_last)},
    {"input": torch.randn(1, 2, 3, 4, 3)},
    {"input": torch.randn(1, 2, 3, 4, 3, 2)},
    {"input": torch.randn(1, 2, 3, 4, 3, 2, 4)},
    {"input": torch.randn(1, 2, 3, 4, 3, 2, 4, 2)},
]


for data in testing.get_raw_data():
    input_data.append({"input": data})


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_frac(input_data, dtype):
    test = testing.OpTest(
        func=torch.frac,
        input_args={"input": input_data["input"].to(dtype)},
    )
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
        test.check_out_ops(fp16=True)
        test.check_grad_fn(fp16=True)
    else:
        test.check_result()
        test.check_out_ops()
        test.check_grad_fn()

    inplace_input = copy.deepcopy(input_data)
    if dtype == torch.float32:
        comparators = testing.DefaultComparator(abs_diff=1e-6, equal_nan=True)
    else:
        comparators = testing.DefaultComparator(
            abs_diff=5e-2, rel_diff=5e-3, equal_nan=True
        )
    test = testing.InplaceOpChek(
        func_name=torch.frac.__name__ + "_",
        self_tensor=inplace_input["input"].to(dtype),
        comparators=[comparators],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", float_dtypes)
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
@pytest.mark.parametrize("dtype", float_dtypes)
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
