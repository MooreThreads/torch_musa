"""Test glu forward & backward operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest

import torch_musa
from torch_musa import testing

# input, other


def elu_forward_inputs():
    return [
        {'inputs': torch.randn((2,), requires_grad=False)},
        {'inputs': torch.randn((32,), requires_grad=False)},
        {'inputs': torch.randn((128,), requires_grad=False)},
        {'inputs': torch.randn((512,), requires_grad=False)},
        {'inputs': torch.randn((1024,), requires_grad=False)},
        {'inputs': torch.randn((16, 1024), requires_grad=False)},
        {'inputs': torch.randn((16, 16, 1024), requires_grad=False)},
        {'inputs': torch.randn((16, 16, 16, 1024), requires_grad=False)},
    ]


support_dtypes = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", elu_forward_inputs())
@pytest.mark.parametrize("dtype", support_dtypes)
def test_elu_fwd(input_data, dtype):
    elu = torch.nn.ELU
    elu_args = {}
    test = testing.OpTest(
        func=elu,
        input_args=elu_args,
        # TODO:(mt-ai) For absolute error, mudnn's res should be set to 1e-6.
        comparators=testing.DefaultComparator(abs_diff=1e-6)
    )
    test.check_result({
        "input": input_data['inputs'].to(dtype),
    }, train=False)
