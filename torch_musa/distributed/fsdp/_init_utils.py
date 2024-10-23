# pylint: disable=all
import itertools
import warnings
import torch
import torch.nn as nn
from typing import (
    Set,
    no_type_check,
    Optional,
    Union,
)

from torch.distributed.fsdp._common_utils import (
    _FSDPDeviceHandle,
    _FSDPState,
)

from torch.distributed.fsdp._init_utils import (
    _get_orig_params,
    _get_modules_to_materialize,
)


@no_type_check
def _init_device_handle(
    state: _FSDPState,
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_id: Optional[Union[int, torch.device]],
) -> _FSDPState:
    """
    Determine device handle used for initializing FSDP.

    If a device is specified by ``device_id``,
    then returns device handle corresponds to that device type. Otherwise, If the
    module is already on a non-CPU device, then the device type is that non-CPU device type.
    If the module is on CPU or meta, then the device type is the current musa device.

    This method will be called once ignored paramters was determined, as the device handle maybe needed
    for other initialization.
    """
    determined_device = None
    if device_id is not None:
        determined_device = (
            device_id
            if isinstance(device_id, torch.device)
            else torch.device(device_id)
        )
    if determined_device is None:
        for param in _get_orig_params(module, ignored_params):
            if param.device.type in {"cpu", "meta"}:
                continue
            if determined_device is None:
                determined_device = param.device
            else:
                if param.device.type != determined_device.type:
                    raise RuntimeError(
                        f"FSDP does not support modules with different device types "
                        f"but got params on {determined_device.type} and {param.device.type}"
                    )
        determined_device = determined_device or torch.device(
            "musa", torch.musa.current_device()
        )

    state._device_handle = _FSDPDeviceHandle.from_device(determined_device)
    return state


def _get_device_from_device_id(
    device_id: Optional[Union[int, torch.device]],
    rank: int,
) -> Optional[torch.device]:
    """
    Return a ``torch.device`` for the specified ``device_id``.

    Processes ``device_id`` and returns either the corresponding device or
    ``None`` if ``device_id`` is ``None``.
    """
    if device_id is None:
        return None
    device = (
        device_id if isinstance(device_id, torch.device) else torch.device(device_id)
    )
    if device == torch.device("musa"):
        warnings.warn(
            f"FSDP got the argument `device_id` {device_id} on rank "
            f"{rank}, which does not have an explicit index. "
            f"FSDP will use the current device {torch.musa.current_device()}. "
            "If this is incorrect, please explicitly call `torch.musa.set_device()` "
            "before FSDP initialization or pass in the explicit device "
            "index as the `device_id` argument."
        )
        device = torch.device("musa", torch.musa.current_device())
    return device


def _materialize_meta_module(
    root_module: nn.Module,
    device_from_device_id: Optional[torch.device],
    ignored_modules: Set[nn.Module],
):
    # Run default meta device initialization
    materialization_device = device_from_device_id or torch.device(
        torch.musa.current_device()
    )
    modules_to_materialize = _get_modules_to_materialize(root_module, ignored_modules)
    try:
        # Assume that each module's `reset_parameters()` only initializes its
        # own parameters and not those of its children
        with torch.no_grad():
            for module in modules_to_materialize:
                # As a contract to the user, only call `reset_parameters()` if
                # the module has directly managed parameters/buffers
                module_state_iter = itertools.chain(
                    module.parameters(recurse=False), module.buffers(recurse=False)
                )
                has_module_states = len(list(module_state_iter)) > 0
                if has_module_states:
                    module.to_empty(device=materialization_device, recurse=False)
                    module.reset_parameters()  # type: ignore[operator]
    except BaseException as e:
        warnings.warn(
            "Unable to call `reset_parameters()` for module on meta "
            f"device with error {str(e)}. Please ensure that your module of"
            f"type {type(module)} implements a `reset_parameters()` method."
        )
        raise e


def _get_compute_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
    rank: int,
) -> torch.device:
    """
    Determine and return this FSDP instance's compute device.

    If a device is
    specified by ``device_id``, then returns that device. Otherwise, If the
    module is already on a non-CPU device, then the compute device is that non-CPU
    device. If the module is on CPU, then the compute device is the current
    device.

    Since this method should be called after materializing the module, any
    non-CPU device should not be meta device. For now, the compute device is
    always a CUDA GPU device with its explicit index.

    Precondition: ``_check_single_device_module()`` and
    ``_move_module_to_device()``.
    """

    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device.type != "cpu":
        compute_device = param.device  # Determined by model param placement
    else:
        if device_from_device_id is not None and device_from_device_id.type != "musa":
            compute_device = device_from_device_id  # Determined by custom backend
        else:
            compute_device = torch.device("musa", torch.musa.current_device())
    if device_from_device_id is not None and compute_device != device_from_device_id:
        raise ValueError(
            f"Inconsistent compute device and `device_id` on rank {rank}: "
            f"{compute_device} vs {device_from_device_id}"
        )
    return compute_device


def _apply_init_utils_patch():
    torch.distributed.fsdp._init_utils._get_device_from_device_id = (
        _get_device_from_device_id
    )
    torch.distributed.fsdp._init_utils._materialize_meta_module = (
        _materialize_meta_module
    )
    torch.distributed.fsdp._init_utils._get_compute_device = _get_compute_device
    torch.distributed.fsdp._init_utils._init_device_handle = _init_device_handle
