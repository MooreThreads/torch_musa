"""Test mvlgamma operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing


test_configs = [
    {"shape": (10,), "p": 1},
    {"shape": (10, 10), "p": 2},
    {"shape": (2, 3, 4, 5), "p": 3},
    {"shape": (100,), "p": 5},
    {"shape": (0,), "p": 1},
    {"shape": (20, 20), "p": 2},
]

input_dtypes = [torch.float32, torch.float16]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", test_configs)
@pytest.mark.parametrize("dtype", input_dtypes)
def test_mvlgamma(config, dtype):
    shape = config["shape"]
    p = config["p"]

    # mvlgamma：input > (p - 1) / 2
    input_tensor = torch.randn(shape, dtype=dtype).abs() + p + 1.0

    input_args = {"input": input_tensor, "p": p}

    if dtype in [torch.float32, torch.bfloat16]:
        comparator = testing.DefaultComparator(abs_diff=1e-5, rel_diff=1e-5)
    elif dtype in [torch.float16]:
        comparator = testing.DefaultComparator(abs_diff=1e-2, rel_diff=1e-2)

    test = testing.OpTest(
        func=torch.mvlgamma,
        input_args=input_args,
        comparators=comparator,
    )

    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()
