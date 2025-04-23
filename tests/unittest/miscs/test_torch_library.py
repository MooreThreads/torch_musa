"""test torch library"""

import pytest

import torch
import torch_musa  # pylint: disable=unused-import


@pytest.mark.parametrize("dispatch_key", ["MUSA", "PrivateUse1"])
def test_musa_dispatch_key(dispatch_key):
    """test new library to register new operators"""
    ns_ = f"{dispatch_key}_lib"
    musa_lib = torch.library.Library(ns_, "DEF")
    musa_lib.define("relu_add_one(Tensor self) -> Tensor")

    def relu_add_one_impl(self):
        return self.relu() + 1.0

    musa_lib.impl("relu_add_one", relu_add_one_impl, dispatch_key)

    x = torch.randn((1024), device="musa")
    out = getattr(torch.ops, ns_).relu_add_one(x)
    assert torch.allclose(out, x.relu() + 1.0)
