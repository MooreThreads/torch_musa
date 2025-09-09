# pylint: disable=missing-module-docstring
import torch
from .device_interface import (
    register_interface_for_device,
    MusaInterface,
    get_registered_device_interfaces,
)
from .variables.torch import *
from .trace_rules import _apply_dynamo_trace_rules_patches
from .backend import musagraphs

_apply_dynamo_trace_rules_patches()

register_interface_for_device("musa", MusaInterface)
for i in range(torch.musa.device_count()):
    register_interface_for_device(f"musa:{i}", MusaInterface)
