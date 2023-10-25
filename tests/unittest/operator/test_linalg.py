"""Test linear algebra operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,not-callable
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    torch.randn(4, 100, 4, 4),
    torch.randn(8, 100, 8, 8),
    torch.randn(16, 100, 16, 16),
]
dim = [0,1,2,3]
order = [1,3]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dim", dim)
@pytest.mark.parametrize("order", order)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_vector_norm(input_data, dim, order, dtype):
    m = torch.linalg.vector_norm
    input_data = input_data.to(dtype)
    output = m(input_data,dim,order)
    musa_input = input_data.to('musa')
    output_musa = m(musa_input,dim,order)
    assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa.cpu())

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", [
    torch.randn(4, 4),
    torch.randn(2, 3, 4, 4)
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_inv(input_data, dtype):
    m = torch.linalg.inv
    input_data = input_data.to(dtype)
    output = m(input_data)
    musa_input = input_data.to('musa')
    output_musa = m(musa_input)
    assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa.cpu())

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", [
    {"A": torch.randn(1,3,3), "B": torch.randn(2,3,3)}
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_lstsq(input_data, dtype):
    m = torch.linalg.lstsq
    input_data["A"] = input_data["A"].to(dtype)
    input_data["B"] = input_data["B"].to(dtype)
    output = m(input_data["A"], input_data["B"])
    output_musa = m(input_data["A"].to('musa'), input_data["B"].to('musa'))
    assert testing.DefaultComparator(abs_diff=1e-5)(output.solution, output_musa.solution.cpu())
    assert testing.DefaultComparator(abs_diff=1e-5)(output.residuals, output_musa.residuals.cpu())
    assert testing.DefaultComparator(abs_diff=1e-5)(output.rank, output_musa.rank.cpu())
    assert testing.DefaultComparator(abs_diff=1e-5)(output.singular_values, \
        output_musa.singular_values.cpu())
