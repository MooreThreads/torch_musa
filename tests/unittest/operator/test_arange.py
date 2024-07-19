"""Test arange operators."""

# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing
import torch_musa

start = [0, 5]
end = [0, 10, 20]
step = [1, 2]
if testing.get_musa_arch() >= 22:
    dtype_list = [
        torch.float32,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
    ]
else:
    dtype_list = [torch.float32, torch.int32, torch.int64, torch.float16]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("start", start)
@pytest.mark.parametrize("end", end)
@pytest.mark.parametrize("step", step)
@pytest.mark.parametrize("dtype", dtype_list)
def test_arange(start, end, step, dtype):
    if start > end:
        pytest.skip("ignore upper and lower bound inconsistent with step cases")

    cpu_res = torch.arange(start=start, end=end, step=step, dtype=dtype, device="cpu")
    musa_res = torch.arange(start=start, end=end, step=step, dtype=dtype, device="musa")

    comparator = testing.DefaultComparator()
    assert comparator(cpu_res.float(), musa_res.float())

    out_res = torch.empty_like(musa_res)
    prev_addr = out_res.data_ptr()

    out_output = torch.arange(start=start, end=end, step=step, dtype=dtype, out=out_res)
    assert prev_addr == out_res.data_ptr()
    assert prev_addr == out_output.data_ptr()
    assert comparator(out_res.float(), out_output.float())
    assert comparator(out_res.float(), musa_res.float())
