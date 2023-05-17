"""Imports the torch musa adaption facilities."""
# pylint: disable=wrong-import-position, W0404

import warnings
import sys
from distutils.version import LooseVersion
import torch

torch_min_version = LooseVersion("2.0.0")
if torch.__version__ < torch_min_version:
    raise RuntimeError(
        "torch version must not be less than v2.0.0 when using torch_musa,",
        " but now torch version is " + torch.__version__)

if '2.0.0' not in torch.__version__:
    warnings.warn(
        'torch version should be v2.0.0 when using torch_musa, but now torch version is ' +
        torch.__version__, UserWarning)

torch.utils.rename_privateuse1_backend("musa")

try:
    import torch_musa._MUSAC
except ImportError as err:
    raise ImportError("Please try running Python from a different directory!") from err

torch.__setattr__('musa', sys.modules[__name__])

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

register_deserialization()


from . import testing

def _sleep(cycles):
    torch_musa._MUSAC._musa_sleep(cycles)
