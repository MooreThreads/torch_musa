# pylint: disable=C0114, C0115, W0611
import warnings

import torch
from torch.amp.grad_scaler import _MultiDeviceReplicator, OptState

from .common import amp_definitely_not_available, _musa_device


class GradScaler(torch.amp.grad_scaler.GradScaler):
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        if enabled and amp_definitely_not_available():
            warnings.warn(
                "torch.musa.amp.GradScaler is enabled, but MUSA is not available.  Disabling."
            )
            enabled = False
        super().__init__(
            device=_musa_device,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
