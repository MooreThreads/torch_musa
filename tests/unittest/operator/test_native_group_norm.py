# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
from torch_musa import testing

import torch_musa


input_dtype = [torch.float32]

parameter = [
    {"data": torch.randn(6, 6), "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(6, 6, 6), "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(20, 6, 10, 10), "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(20, 6, 10, 1, 10), "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(20, 6, 10, 2, 1, 10), "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(20, 6, 10, 10, 1, 2, 3), "num_groups": 3, "num_channels": 6},
    {
        "data": torch.randn(20, 6, 10, 10, 1, 2, 3, 1),
        "num_groups": 3,
        "num_channels": 6,
    },
]

affine = [True]

eps = [1e-5, 0, 0.5]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_dtype", input_dtype)
@pytest.mark.parametrize("parameter", parameter)
@pytest.mark.parametrize("affine", affine)
@pytest.mark.parametrize("eps", eps)
def test_native_layer_norm(input_dtype, parameter, affine, eps):
    test = testing.OpTest(
        func=torch.nn.GroupNorm,
        input_args={
            "num_groups": parameter["num_groups"],
            "num_channels": parameter["num_channels"],
            "eps": eps,
            "affine": affine,
        },
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result(
        inputs={
            "input": torch.tensor(
                parameter["data"].clone().detach(), dtype=input_dtype, requires_grad=True
            )
        },
        train=True,
    )
