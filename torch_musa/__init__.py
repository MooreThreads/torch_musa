"""Imports the torch musa adaption facilities."""
# pylint: disable=wrong-import-position

import torch

torch.utils.rename_privateuse1_backend("musa")

from .core.device import Device as device
from .core.device import (
    set_device,
    current_device,
    is_available,
    device_count,
    synchronize,
)

from .core.serialization import register_deserialization

register_deserialization()

try:
    import torch_musa._MUSAC
except ImportError as err:
    raise ImportError("Please try running Python from a different directory!") from err

from . import testing
