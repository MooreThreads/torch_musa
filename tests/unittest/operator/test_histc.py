"""Test histc operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import pytest
import torch
from torch_musa import testing


# CUDA supports int8, int16, int32, int64, float32, float64
dtypes = [
    torch.float32,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"input": torch.randint(0, 20, (3,))},
        {"input": torch.randint(0, 20, (10,))},
        {"input": torch.randint(0, 20, (20,))},
        {"input": torch.randint(0, 20, (20, 5))},
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("bins", [5, 10, 20])
def test_histc(input_data, dtype, bins):
    vmin, vmax = 0, 20
    # gen cpu res, and cpu only supports fp32
    cpu_input = input_data["input"].clone().to(torch.float32)
    cpu_out = torch.histc(cpu_input, bins, vmin, vmax)
    # gen musa res
    musa_input = input_data["input"].clone().musa().to(dtype)
    musa_out = torch.histc(musa_input, bins, vmin, vmax)
    test = testing.OpTest(
        func=torch.histc,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.compare_res(cpu_out, musa_out.float().cpu())
