"""unittest for channel_shuffle operator in MUSA"""

# pylint: disable=missing-function-docstring,redefined-outer-name,redefined-builtin,unused-import

import pytest
import torch
import torch_musa
from torch_musa import testing


shapes = [
    (1, 4, 2, 2),
    (2, 8, 4, 4),
    (4, 12, 8, 8),
    (8, 12, 64, 64),
]
groups = [2, 4]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("groups", groups)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_channel_shuffle_forward(shape, groups, dtype):
    input = {
        "input": torch.randn(shape, dtype=dtype, requires_grad=True),
        "groups": groups,
    }

    test = testing.OpTest(
        func=torch.channel_shuffle,
        input_args=input,
        comparators=testing.DefaultComparator(
            abs_diff=1e-5 if dtype == torch.float32 else 1e-3,
            rel_diff=1e-5 if dtype == torch.float32 else 1e-3,
        ),
    )
    test.check_result()
    test.check_result(train=True)
