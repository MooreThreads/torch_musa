"""Test isin operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_shapes = [
    [],
    [10],
    [16, 32],
    [32, 128, 20],
    [10, 1, 45, 3],
]

test_shapes = [
    [],
    [2],
    [2, 3],
    [3, 3, 4],
]

dtypes = testing.get_all_support_types()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("test_shape", test_shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("invert", [True, False])
def test_isin(input_shape, test_shape, dtype, invert):
    input_data = torch.randint(0, 100, input_shape).to(dtype)
    test_data = torch.randint(0, 100, test_shape).to(dtype)
    args = {
        "elements": input_data,
        "test_elements": test_data,
        "invert": invert,
    }
    test = testing.OpTest(
        func=torch.isin,
        input_args=args,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()
