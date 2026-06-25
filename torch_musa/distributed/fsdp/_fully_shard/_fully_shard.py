"""patch to add lazy_hsdp_allreduce switch"""
from typing import cast

import torch
from torch import nn

from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule

def set_lazy_hsdp_allreduce(self, lazy_hsdp_allreduce: bool, *, recurse: bool = True) -> None:
    """
    Sets if the module should use lazy HSDP all-reduce. This can be used to
    implement lazy HSDP all-reduce for FSDP modules.
    """
    self_module = cast(nn.Module, self)
    modules = list(self_module.modules()) if recurse else [self_module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.lazy_hsdp_allreduce = lazy_hsdp_allreduce

def _apply_fully_shard_class_patch():
    (torch.distributed.fsdp._fully_shard._fully_shard.FSDPModule
    .set_lazy_hsdp_allreduce) = set_lazy_hsdp_allreduce
