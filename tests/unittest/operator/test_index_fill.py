"""Test index_fill operator."""

import torch
import pytest
from torch_musa import testing

# fmt: off
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config",
    [
        # self_shape, dim, index_numel
        [[16, 1024], 0, 0],
        [[16, 1024], 0, 7],
        [[16, 1023], 0, 12],
        [[16, 128, 2], 1, 64],
        [[16, 128, 2], 2, 4],
        [[2, 4, 16, 8, 2, 16], 2, 8],
    ],
)
@pytest.mark.parametrize("dtype", [
    torch.float32, torch.half, torch.bfloat16, torch.bool,
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
])
def test_index_fill(config, dtype):
    """test index_fill with supported dtypes"""
    fill_dim, index_numel = config[1], config[2]
    input_data = {
        "input": torch.ones(config[0], dtype=dtype),
        "dim": fill_dim,
        "index": torch.randint(
            0, config[0][fill_dim], (index_numel,), dtype=torch.int64
        ),
        "value": 2.0
    }
    if dtype == torch.float16:
        abs_diff, rel_diff = (1e-2, 5e-4)
    elif dtype == torch.bfloat16:
        abs_diff, rel_diff = (5e-2, 5e-3)
    else:
        abs_diff, rel_diff = (1e-6, 1e-6)

    comparator = testing.DefaultComparator(abs_diff, rel_diff)
    test = testing.OpTest(func=torch.index_fill, input_args=input_data, comparators=comparator)
    test.check_result()
