# pylint: disable=missing-module-docstring,import-outside-toplevel
import functools

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


def _apply_dynamo_reset_patch():
    """Also clear musagraph trees from torch._dynamo.reset."""
    if getattr(torch._dynamo.reset, "_torch_musa_patched", False):
        return

    original_reset = torch._dynamo.reset

    @functools.wraps(original_reset)
    def reset() -> None:
        from torch._dynamo import convert_frame

        from torch_musa._inductor.musagraph_trees import reset_musagraph_trees

        with convert_frame.compile_lock:
            original_reset()
            reset_musagraph_trees()

    reset._torch_musa_patched = True
    torch._dynamo.reset = reset


_apply_dynamo_reset_patch()

register_interface_for_device("musa", MusaInterface)
for i in range(torch.musa.device_count()):
    register_interface_for_device(f"musa:{i}", MusaInterface)
