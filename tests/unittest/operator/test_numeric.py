"""Test numeric operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, unexpected-keyword-arg, missing-module-docstring
import pytest
import torch
from torch_musa import testing

input_shapes = [
    (4, 6),
    (128, 128),
    (1024, 1024),
]


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("shape", input_shapes)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_trace(shape, dtype):
    device = "musa"
    input_data = torch.randn(shape, dtype=dtype).to(device)
    test = testing.OpTest(
        func=torch.trace,
        input_args={"input": input_data},
        comparators=testing.DefaultComparator(),
    )
    test.check_result()
