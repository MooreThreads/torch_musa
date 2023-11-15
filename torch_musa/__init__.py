"""Imports the torch musa adaption facilities."""
# pylint: disable=wrong-import-position, W0404, C0103

import warnings
import sys
from typing import Set, Type
from packaging.version import Version
import torch
try:
    from .version import __version__
except ImportError:
    pass

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

_tensor_classes: Set[Type] = set()

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
from .core import amp
from .core.amp.common import (
    amp_definitely_not_available,
    get_amp_supported_dtype,
    is_autocast_musa_enabled,
    is_autocast_cache_enabled,
    set_autocast_musa_enabled,
    set_autocast_musa_dtype,
    get_autocast_musa_dtype,
    set_autocast_cache_enabled,
    clear_autocast_cache,
    autocast_increment_nesting,
    autocast_decrement_nesting,
)

from .core.serialization import register_deserialization

from .core.memory import (
    set_per_process_memory_fraction,
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
    mem_get_info,
)


from .core._lazy_init import _lazy_init

from .core.random import *

from .core.mudnn import *

# A hack to get `torch.backends.mudnn` functions/attributes. This allows users to use cudnn
# equivalent functions like `torch.backends.mudnn.allow_tf32 = True`
torch.backends.__setattr__("mudnn", sys.modules["torch_musa.core.mudnn"])  # pylint: disable=C2801

register_deserialization()


def _sleep(cycles):
    torch_musa._MUSAC._musa_sleep(cycles)


setattr(torch.version, "musa", torch_musa._MUSAC._musa_version)

from .core.tensor_attrs import set_torch_attributes
from .core.module_attrs import set_module_attributes

def set_attributes():
    """Set attributes for torch."""
    set_torch_attributes()
    set_module_attributes()

set_attributes()
