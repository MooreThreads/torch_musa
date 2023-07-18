"""Test tril operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

data_type = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("data_type", data_type)
def test_multinomial(data_type):
    input_data = torch.tensor([0, 10, 3, 0], device="musa").to(data_type)
    num_samples = 2
    replacement = False
    result = torch.multinomial(input=input_data,
                               num_samples=num_samples,
                               replacement=replacement).cpu().numpy()
    assert len(result) == 2
    assert result[0] == 1 or result[0] == 2
    assert result[1] == 1 or result[1] == 2

    num_samples = 4
    replacement = True
    result = torch.multinomial(input=input_data,
                               num_samples=num_samples,
                               replacement=replacement).cpu().numpy()
    assert len(result) == 4
    assert result[0] == 1 or result[0] == 2
    assert result[1] == 1 or result[1] == 2
    assert result[2] == 1 or result[2] == 2
    assert result[3] == 1 or result[3] == 2

    input_data = torch.tensor([1, 10, 3, 1], device="musa").to(data_type)
    num_samples = 4
    replacement = False
    result = torch.multinomial(input=input_data,
                               num_samples=num_samples,
                               replacement=replacement).cpu().numpy()
    assert len(result) == 4
    assert result[0] == 0 or result[0] == 1 or result[0] == 2 or result[0] == 3
    assert result[1] == 0 or result[1] == 1 or result[1] == 2 or result[1] == 3
    assert result[2] == 0 or result[2] == 1 or result[2] == 2 or result[2] == 3
    assert result[3] == 0 or result[3] == 1 or result[3] == 2 or result[3] == 3
