"""Test masked operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, unexpected-keyword-arg
import pytest
import torch
import torch_musa
from torch_musa import testing


def generate_mask(shape):
    mask = torch.randint(0, 2, shape).bool()
    return mask


dtypes = testing.get_all_support_types()
dtypes.extend([torch.uint8, torch.int16, torch.float16, torch.float64, torch.bool])
if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


input_data = [
    torch.randn(10),
    torch.randn(16, 32),
    torch.randn(16, 32).T,
    torch.randn(32, 8, 16),
    torch.randn(1, 16, 7, 33),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", dtypes)
def test_masked_select(input_data, dtype):
    if testing.get_musa_arch() < 22 and dtype == torch.bfloat16:
        return
    data = input_data.to(dtype)
    mask = generate_mask(data.shape)
    test = testing.OpTest(
        func=torch.masked_select, input_args={"input": data, "mask": mask}
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", dtypes)
def test_masked_scatter(input_data, dtype):
    if testing.get_musa_arch() < 22 and dtype == torch.bfloat16:
        return
    data = input_data.to(dtype)
    mask = generate_mask(data.shape)
    source = torch.randn(data.shape).to(dtype)

    mu_data = data.musa()
    mu_mask = mask.musa()
    mu_source = source.musa()
    golden = data.masked_scatter(mask, source)
    result = mu_data.masked_scatter(mu_mask, mu_source)
    torch.allclose(golden, result.cpu())
