# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing


data_type = [torch.float32]
#data_type = [torch.float32, torch.int32, torch.int64]
def get_abs_inputs():
    return [
        {"input": torch.tensor(5.0)},
        {"input": torch.randn(1)},
        {"input": torch.randn(1, 2)},
        {"input": torch.randn(1, 0)},
        {"input": torch.randn(1, 0)},
        {"input": torch.randn(1, 1)},
        {"input": torch.randn(1, 2, 3)},
        {"input": torch.randn(1, 0, 3)},
        {"input": torch.randn(1, 2, 3, 4)},
        {"input": torch.randn(1, 2, 6, 3, 4)},
        {"input": torch.randn(1, 2, 6, 3, 4, 2)},
        {"input": torch.randn(1, 2, 6, 3, 4, 2, 8)},
        {"input": torch.randn(1, 2, 6, 3, 4, 2, 8, 2)},
    ]


@pytest.mark.parametrize("input_args", get_abs_inputs())
@pytest.mark.parametrize("data_type", data_type)
def test_abs(input_args, data_type):
    test = testing.OpTest(
        func=torch.abs,
        input_args={"input": input_args["input"].to(data_type)}
    )
    test.check_result(None)
