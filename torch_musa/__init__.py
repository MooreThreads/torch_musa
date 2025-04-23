"""Imports the torch musa adaption facilities."""

# pylint: disable=wrong-import-position, W0404, C0103, C2801

import sys
import warnings
import importlib
from typing import Set, Type

import torch
from torch.distributed.fsdp import (
    sharded_grad_scaler,
)  # load sharded_grad_scaler explicitly
from packaging.version import Version

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

if "2.2.0" not in torch.__version__:
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

torch.__setattr__("musa", sys.modules[__name__])

# Hack torch profiler with our torch_musa version
sys.modules["torch.autograd.profiler"] = importlib.import_module(
    "torch_musa.autograd.profiler"
)
torch.autograd.__setattr__("profiler", sys.modules["torch.autograd.profiler"])

sys.modules["torch.autograd.profiler_util"] = importlib.import_module(
    "torch_musa.autograd.profiler_util"
)
torch.autograd.__setattr__("profiler_util", sys.modules["torch.autograd.profiler_util"])

sys.modules["torch.profiler"] = importlib.import_module("torch_musa.profiler")
torch.__setattr__("profiler", sys.modules["torch.profiler"])

setattr(torch._C._autograd.DeviceType, "MUSA", 1)
setattr(torch._C._profiler.ProfilerActivity, "MUSA", 1)

from torch_musa.distributed import _apply_distributed_patch

from .core.device import (
    Device as device,
    DeviceOf as device_of,
    set_device,
    set_default_dtype,
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
    is_autocast_enabled,
    set_autocast_enabled,
    set_autocast_dtype,
    get_autocast_dtype,
)

from .core.serialization import register_deserialization

from .core import memory
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
    reset_peak_memory_stats,
    _record_memory_history,
    _dump_snapshot,
)

from .core._lazy_init import _lazy_init, _initialized, _is_in_bad_fork

from .core.random import *

torch.random.fork_rng = fork_rng

from .core.mudnn import *

# A hack to get `torch.backends.mudnn` functions/attributes. This allows users to use cudnn
# equivalent functions like `torch.backends.mudnn.allow_tf32 = True`
torch.backends.__setattr__("mudnn", sys.modules["torch_musa.core.mudnn"])

register_deserialization()

# A hack to get `torch.set_default_tensor_type` and `torch.set_default_dtype`
torch.set_default_dtype = set_default_dtype


def _sleep(cycles):
    torch_musa._MUSAC._musa_sleep(cycles)


setattr(torch.version, "musa", torch_musa._MUSAC._musa_version)

from .core.tensor_attrs import set_torch_attributes
from .core.module_attrs import set_module_attributes
from .core.storage import set_storage_attributes

from .core.storage import set_storage_attributes


def set_attributes():
    """Set attributes for torch."""
    set_torch_attributes()
    set_module_attributes()
    set_storage_attributes()


set_attributes()


def _apply_storage_patch():
    # keep consistent with CUDA
    # releated PR: https://github.com/pytorch/pytorch/pull/99882
    def untyped_storage_resize_(self, size):
        torch_musa._MUSAC._musa_storage_resize_(self, size)

    # TODO: check code on pybind side
    # torch.UntypedStorage.resize_ = torch_musa._MUSAC._musa_storage_resize_
    torch.UntypedStorage.resize_ = untyped_storage_resize_


def _apply_sharded_grad_scaler_patch():
    torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler = (
        torch_musa.distributed.fsdp.ShardedGradScaler
    )


def _apply_patches():
    _apply_distributed_patch()
    _apply_storage_patch()
    _apply_sharded_grad_scaler_patch()


_apply_patches()


def ipc_collect():
    _lazy_init()
    return _MUSAC._musa_ipc_collect()


from .core.reductions import init_reductions

init_reductions()

from . import _dynamo

# for the purpose of lazily import inductor related modules
from ._inductor import _init_inductor_backend_registration  # internal usage


def overwrite_cuda_api():
    torch.cuda._lazy_init = _lazy_init


overwrite_cuda_api()

from .core.ao import *

from .core.ao.register_helper import _register_ao_intrinsic_modules

_register_ao_intrinsic_modules()


# TODO: more elegant patching
setattr(
    torch._C,
    "_conv_determine_backend_memory_format",
    torch_musa._MUSAC._conv_determine_backend_memory_format,
)
