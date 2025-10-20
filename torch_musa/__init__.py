"""Imports the torch musa adaption facilities."""

# pylint: disable=wrong-import-position, W0404, C0103, C2801, W0602

import sys
import warnings
import importlib
import os
import re
from typing import Set, Type
import contextlib

from packaging.version import Version

try:
    from .version import __version__
except ImportError:
    pass

is_loaded = False


def _autoload():
    global is_loaded

    if is_loaded:
        print("torch_musa already loaded.")
        return


import torch

TORCH_MIN_VERSION = Version("2.5.0")
TORCH_VERSION = Version(torch.__version__).base_version
if Version(TORCH_VERSION) < TORCH_MIN_VERSION:
    raise RuntimeError(
        "torch version must not be less than v2.5.0 when using torch_musa,",
        " but now torch version is " + torch.__version__,
    )

if "2.5.0" not in torch.__version__:
    warnings.warn(
        "torch version should be v2.5.0 when using torch_musa, but now torch version is "
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

setattr(
    torch._C._profiler.ProfilerActivity,
    "MUSA",
    torch._C._profiler.ProfilerActivity.PrivateUse1,
)

from torch_musa.distributed import _apply_distributed_patch

from .core import (
    current_blas_handle,
    current_blaslt_handle,
)
from .core.device import (
    Device as device,
    DeviceOf as device_of,
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
    _maybe_exchange_device,
    _DeviceGuard,
    register_musa_hook,
    get_arch_list,
    is_bf16_supported,
)

register_musa_hook()

from .core.stream import (
    _set_stream_by_id,
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
from .core.memory import *

from .core._lazy_init import _lazy_init, _initialized, _is_in_bad_fork, is_initialized

from .core.random import *

torch.random.fork_rng = fork_rng

from .core.mudnn import *
from .core.ops import *

from .musa_graph import *

from .core import mccl

# A hack to get `torch.backends.mudnn` functions/attributes. This allows users to use cudnn
# equivalent functions like `torch.backends.mudnn.allow_tf32 = True`
torch.backends.__setattr__("mudnn", sys.modules["torch_musa.core.mudnn"])

register_deserialization()


def _sleep(cycles):
    torch_musa._MUSAC._musa_sleep(cycles)


setattr(torch.version, "musa", torch_musa._MUSAC._musa_version)

from .core.tensor_attrs import set_torch_attributes
from .core.module_attrs import set_module_attributes
from .core.storage import set_storage_attributes

from .core.storage import set_storage_attributes

from .profiler import set_profiler_attributes


def set_attributes():
    """Set attributes for torch."""
    set_torch_attributes()
    set_module_attributes()
    set_storage_attributes()
    set_profiler_attributes()


set_attributes()


def _apply_sharded_grad_scaler_patch():
    # avoid weird issue (torch.musa.is_available() is False in triton/backends/mtgpu/driver.py)
    # which caused by circular import when using triton.
    # pylint: disable=import-outside-toplevel, unused-import
    from torch.distributed.fsdp import sharded_grad_scaler

    torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler = (
        torch_musa.distributed.fsdp.ShardedGradScaler
    )


def _apply_patches():
    _apply_distributed_patch()
    _apply_sharded_grad_scaler_patch()


_apply_patches()


def ipc_collect():
    _lazy_init()
    return _MUSAC._musa_ipc_collect()


from .core.reductions import init_reductions

init_reductions()

# Avoiding load all modules of torch._inductor before registing our
# inductor backend, but things works not well if using the registration way of
# custom device that provided by PyTorch, thus we use `_init_inductor_backend_registration`
# as interface for custom device registration.
# see patch torch_patches/torch__inductor_codegen_common.py.patch
from ._dynamo import *
from ._inductor import _init_inductor_backend_registration  # for internal usage


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

OutOfMemoryError = torch._C.OutOfMemoryError

# pylint: disable=C0411, C0412
import torch._inductor.graph as orig_graph
import torch_musa._inductor.graph as musa_graph

setattr(orig_graph, "GraphLowering", musa_graph.MusaGraphLowering)


# pylint: disable=missing-function-docstring
def parseEnvForUMM():
    val = os.getenv("PYTORCH_MUSA_ALLOC_CONF")
    os.environ["CPU_UNIFIED_FLAG"] = "False"  # inner flag, not exposed to user
    if val is not None:
        config = val
        options = re.split(r"[\s,]+", config.strip())
        for option in options:
            kv = re.split(r"[:]+", option.strip())
            if len(kv) == 2:
                if kv[0] == "cpu":
                    if kv[1] == "unified":
                        torch_musa._MUSAC._musa_register_unified_cpu_allocator()
                        os.environ["CPU_UNIFIED_FLAG"] = "True"
                        return


parseEnvForUMM()


@contextlib.contextmanager
def use_unified_cpu_allocator():
    r"""A context manager that use unified cpu allocator.

    Args:
      None

    .. note::
        unified cpu allocator works in context
    """
    torch_musa._MUSAC._musa_register_unified_cpu_allocator()
    os.environ["CPU_UNIFIED_FLAG"] = "True"
    try:
        yield
    finally:
        torch_musa._MUSAC._musa_reset_unified_cpu_allocator()
        os.environ["CPU_UNIFIED_FLAG"] = "False"
