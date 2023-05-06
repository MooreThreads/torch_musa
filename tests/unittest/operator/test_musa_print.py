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

@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", types)
def test_musa_print(input_data, dtype):
    captured_out = io.StringIO()
    sys.stdout = captured_out
    print(input_data.to(dtype).to("musa"))
    sys.stdout = sys.__stdout__
    assert captured_out.getvalue() == str(input_data.to(dtype).to("musa")) + "\n"
