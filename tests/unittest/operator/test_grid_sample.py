"""Test grid_sample operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from functools import partial
import pytest
import torch
import torch.nn.functional as F
from torch_musa import testing

input_datas = [
    # FIXME(yueran.tang) Fix the following tests due to GPU fault.
    # {"tenInput": torch.randn([2, 3, 2176, 3840], requires_grad=True),
    #  "tenFlow": torch.randn([2, 2176, 3840, 2], requires_grad=True)},
    # {"tenInput": torch.randn([3, 7, 512, 256], requires_grad=True),
    #  "tenFlow": torch.randn([3, 512, 256, 2], requires_grad=True)},
    # {"tenInput": torch.randn([5, 10, 32, 32], requires_grad=True),
    #  "tenFlow": torch.randn([5, 32, 32, 2], requires_grad=True)},
    # {"tenInput": torch.randn([11, 1, 983, 437], requires_grad=True),
    #  "tenFlow": torch.randn([11, 983, 437, 2], requires_grad=True)},
    {"tenInput": torch.randn([17, 16, 16, 16], requires_grad=True),
     "tenFlow": torch.randn([17, 16, 16, 2], requires_grad=True)},
]

all_support_mode = ["bilinear", "nearest"]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("mode", all_support_mode)
def test_func(input_data, mode):
    test = testing.OpTest(
              func=F.grid_sample,
              input_args={"input": input_data["tenInput"],
                          "grid": input_data["tenFlow"],
                          "mode": mode,
                          "padding_mode": "border",
                          "align_corners": True})
    test.check_result(train=False)
