"""Test lazy registration of musa backend of TorchInductor

This unittest should be executed first under unittest/inductor test cases,
so "aaa" is used as the filename's prefix.
"""

# pylint: disable=missing-function-docstring, unused-import

import pytest
import torch
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
)
from torch._inductor.graph import GraphLowering
from torch_musa.testing.base_test_tool import _HAS_TRITON

import torch_musa


def func():
    x, y = 10, 20
    return x + y


# Run this test first, pytest-ordering not installed
# @pytest.mark.run(order=0)
@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
def test_inductor_registration():
    scheduling = get_scheduling_for_device("musa")
    wrapper_codegen = get_wrapper_codegen_for_device("musa")

    assert scheduling is None and wrapper_codegen is None

    GraphLowering(gm=torch.fx.symbolic_trace(func))
    scheduling = get_scheduling_for_device("musa")
    wrapper_codegen = get_wrapper_codegen_for_device("musa")

    assert scheduling and wrapper_codegen
