# pylint: disable= missing-module-docstring, missing-class-docstring,missing-function-docstring,unused-import,unused-variable,not-callable
import os
import random
import torch
from torch import nn
import numpy as np
import torch_musa
from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_amp_autocast():
    scale = torch.tensor([65536.0], device="musa")
    growth_tracker = torch.tensor([0], device="musa", dtype=torch.int32)
    found_inf_combined = torch.tensor([1.0], device="musa")
    growth_factor = 2.0
    backoff_factor = 0.5
    growth_interval = 2000

    torch._amp_update_scale_(
        scale,
        growth_tracker,
        found_inf_combined,
        growth_factor,
        backoff_factor,
        growth_interval,
    )
    testing.DefaultComparator(scale, scale / 2)

    found_inf_combined = torch.tensor([0.0], device="musa")
    torch._amp_update_scale_(
        scale,
        growth_tracker,
        found_inf_combined,
        growth_factor,
        backoff_factor,
        growth_interval,
    )
    testing.DefaultComparator(scale, scale)
