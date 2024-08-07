"""Test musa print."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import io
import sys
import numpy as np
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = testing.get_raw_data()
types = testing.get_all_support_types()


# TODO(MT-AI): fix error when testing on GPU 1, especially for torch.randn(10, 10, 2, 2, 1, 3, 2, 2)
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", types)
def test_musa_print(input_data, dtype):
    captured_out = io.StringIO()
    sys.stdout = captured_out
    print(input_data.to(dtype).to("musa"))
    sys.stdout = sys.__stdout__
    assert captured_out.getvalue() == str(input_data.to(dtype).to("musa")) + "\n"


def test_allow_tf32_get_set_with():
    with torch.backends.mudnn.flags(allow_tf32=False):
        assert torch.backends.mudnn.allow_tf32 is False
    with torch.backends.cudnn.flags(allow_tf32=True):
        assert torch.backends.cudnn.allow_tf32


def test_allow_tf32_get_set():
    orig = torch.backends.mudnn.allow_tf32
    assert torch_musa._MUSAC._get_allow_tf32() == orig

    torch.backends.mudnn.allow_tf32 = not orig
    assert torch_musa._MUSAC._get_allow_tf32() == (not orig)
    torch.backends.mudnn.allow_tf32 = orig
