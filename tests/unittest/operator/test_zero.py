"""Test zero operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape",
[
    (1, 16),
    (2, 4),
    (4, 32, 32),
    (3, 32, 32, 2),
    (256, 32, 32, 2, 2),
    (4, 6, 8, 2, 4, 8),
    (2, 4, 6, 8, 2, 4, 8),
])
@pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
def test_fill(input_shape, data_type):
    if testing.get_musa_arch() < 22:
        return
    test = testing.OpTest(
        func=torch.zeros,
        input_args={
            "size": input_shape,
            "dtype": data_type
        },
    )
    test.check_result()
