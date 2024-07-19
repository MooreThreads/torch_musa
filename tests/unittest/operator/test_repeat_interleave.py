"""Test repeat interleave operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
from torch import nn
import pytest
import torch_musa

from torch_musa import testing

# FIXME(yuerang.tang) We skip several large inputs for now to avoid the OOM error.
input_data = testing.get_raw_data()[:3]

# not support for fp16
support_dtypes = [torch.float32, torch.int32, torch.int64]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_repeat_interleave(input_data, dtype):
    input_data = input_data.to(dtype)
    output_cpu = input_data.repeat_interleave(2)
    output_musa = input_data.to("musa").repeat_interleave(2)
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")

    output_cpu = torch.repeat_interleave(input_data, 2)
    output_musa = torch.repeat_interleave(input_data.to("musa"), 2)
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")

    output_cpu = torch.repeat_interleave(input_data, 3, dim=0)
    output_musa = torch.repeat_interleave(input_data.to("musa"), 3, dim=0)
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")

    repeat_tensor = torch.randint(1, 2, (input_data.size()[0],))
    output_cpu = torch.repeat_interleave(input_data, repeat_tensor, dim=0)
    output_musa = torch.repeat_interleave(
        input_data.to("musa"), repeat_tensor.to("musa"), dim=0
    )
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")

    output_size = sum(repeat_tensor)
    output_cpu = torch.repeat_interleave(
        input_data, repeat_tensor, dim=0, output_size=output_size
    )
    output_musa = torch.repeat_interleave(
        input_data.to("musa"), repeat_tensor.to("musa"), dim=0, output_size=output_size
    )
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")
