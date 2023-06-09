"""Imports the torch musa adaption facilities."""
# pylint: disable=wrong-import-position, W0404, C0103

import warnings
import sys
from packaging.version import Version
import torch

TORCH_MIN_VERSION = Version("2.0.0")
TORCH_VERSION = Version(torch.__version__).base_version
if Version(TORCH_VERSION) < TORCH_MIN_VERSION:
    raise RuntimeError(
        "torch version must not be less than v2.0.0 when using torch_musa,",
        " but now torch version is " + torch.__version__,
    )

if "2.0.0" not in torch.__version__:
    warnings.warn(
        "torch version should be v2.0.0 when using torch_musa, but now torch version is "
        + torch.__version__,
        UserWarning,
    )

torch.utils.rename_privateuse1_backend("musa")

try:
    import torch_musa._MUSAC
except ImportError as err:
    raise ImportError("Please try running Python from a different directory!") from err

torch.__setattr__("musa", sys.modules[__name__])  # pylint: disable=C2801


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
    _DeviceGuard,
)

from .core.stream import (
    set_stream,
    current_stream,
    default_stream,
    ExternalStream,
    stream,
    Stream,
    Event,
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
    max_memory_reserved,
)


from .core._lazy_init import _lazy_init

from .core.random import *

register_deserialization()


def _sleep(cycles):
    torch_musa._MUSAC._musa_sleep(cycles)


from . import testing
