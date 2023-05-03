"""Imports the torch musa adaption facilities."""
# pylint: disable=wrong-import-position, W0404

import torch

torch.utils.rename_privateuse1_backend("musa")

from .core.device import Device as device
from .core.device import DeviceOf as device_of
from .core.device import (
    set_device,
    current_device,
    is_available,
    device_count,
    synchronize,
    get_device_name,
    get_device_capability,
    get_device_properties,
    can_device_access_peer,
    _exchange_device,
)

from .core.stream import (
    set_stream,
    current_stream,
    default_stream,
    stream,
    ExternalStream,
    stream,
    Stream,
)

from .core.serialization import register_deserialization

from .core.memory import (
    empty_cache,
    reset_peak_stats,
    memory_stats,
    memory_summary,
    memory_stats_all,
    memory_reserved,
    memory_snapshot,
    memory_allocated,
    max_memory_allocated,
    max_memory_reserved)

register_deserialization()

try:
    import torch_musa._MUSAC
except ImportError as err:
    raise ImportError("Please try running Python from a different directory!") from err

from . import testing
