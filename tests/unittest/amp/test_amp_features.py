# pylint: disable= missing-module-docstring, missing-class-docstring,missing-function-docstring,unused-import,unused-variable,not-callable
import os
import random
import pytest
import torch
from torch import nn
import numpy as np
import torch_musa
from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_amp_autocast_enabled():
    model = nn.Linear(5, 4)
    model = model.to("musa")
    autocast = torch.musa.amp.autocast
    with autocast(
        enabled=True,
    ):
        input_tensor = torch.randn(3, 5).to("musa")
        output = model(input_tensor)
        assert output.dtype == torch.float16
    output = model(input_tensor)
    assert output.dtype == torch.float32


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_amp_autocast_disabled():
    model = nn.Linear(5, 4)
    model = model.to("musa")
    autocast = torch.musa.amp.autocast
    with autocast(
        enabled=False,
    ):
        input_tensor = torch.randn(3, 5).to("musa")
        output = model(input_tensor)
        # not cast
        assert output.dtype == torch.float32


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_amp_autocast_fp32():
    model = nn.Linear(5, 4)
    model = model.to("musa")
    autocast = torch.musa.amp.autocast
    with autocast(dtype=torch.float32):
        input_tensor = torch.randn(3, 5).to("musa")
        output = model(input_tensor)
        # not cast
        assert output.dtype == torch.float32


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_autocast_fp16_device():
    model = nn.Linear(5, 4)
    model = model.to("musa")
    autocast = torch.autocast
    with autocast(dtype=torch.float16, device_type="musa"):
        input_tensor = torch.randn(3, 5).to("musa")
        output = model(input_tensor)
        # not cast
        assert output.dtype == torch.float16


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_autocast_fp32_device():
    model = nn.Linear(5, 4)
    model = model.to("musa")
    autocast = torch.autocast
    with autocast(dtype=torch.float32, device_type="musa"):
        input_tensor = torch.randn(3, 5).to("musa")
        output = model(input_tensor)
        # not cast
        assert output.dtype == torch.float32


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_autocast_bf16_device():
    model = nn.Linear(5, 4)
    model = model.to("musa")
    autocast = torch.autocast
    with autocast(dtype=torch.bfloat16, device_type="musa"):
        input_tensor = torch.randn(3, 5).to("musa")
        output = model(input_tensor)
        # not cast
        assert output.dtype == torch.bfloat16


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_get_amp_supported_dtype():
    support_dtype = torch.musa.get_amp_supported_dtype()
    assert support_dtype == [torch.float16, torch.bfloat16, torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("enabled", [True, False])
def test_set_autocast_musa_enabled(enabled):
    torch.musa.set_autocast_enabled(enabled)
    res = torch.musa.is_autocast_enabled()
    assert res == enabled
