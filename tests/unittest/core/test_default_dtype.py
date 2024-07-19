"""Test set default dtype"""

# pylint: disable=missing-function-docstring,missing-module-docstring,redefined-outer-name,unused-import
import pytest
import torch
import torch_musa
from torch_musa import testing


dtypes = [
    torch.float,
    torch.double,
    torch.float16,
    torch.bfloat16,
]


@testing.skip_if_musa_unavailable
@pytest.mark.parametrize("dtype", dtypes)
def test_set_default_dtype(dtype):
    torch.set_default_dtype(dtype)
    tensor = torch.randn(10).musa()
    assert tensor.dtype == dtype
    # reset default type to avoid UT failing
    torch.set_default_dtype(torch.float)
    torch.musa.empty_cache()
