"""utilities for testing FSDP"""

# pylint: disable=invalid-name,unused-import
from typing import Tuple

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import Shard

from .common_dist import (
    MultiProcessingTest,
    skip_if_lt_x_gpu,
)


class FSDPTest(MultiProcessingTest):
    """class FSDPTest"""

    def _device_mesh_plan(self, device="musa", hybrid_shard=False):
        if hybrid_shard:
            assert self.world_size % 2 == 0
            replica_group_size = 2
            sharding_group_size = self.world_size // 2
            device_mesh = init_device_mesh(
                device,
                (replica_group_size, sharding_group_size),
                mesh_dim_names=("dp_replicate", "dp_shard"),
            )
        else:
            device_mesh = init_device_mesh(
                device, (self.world_size,), mesh_dim_names=("dp_replicate",)
            )
        return device_mesh


# pylint: disable=unused-argument
def check_sharded_parity(
    cls,
    replicated_module: torch.nn.Module,
    sharded_module: torch.nn.Module,
    prefixes_to_ignore: Tuple[str, ...] = (),
):
    """check the parity between sharded_module and replicated_module"""
    for (replicated_name, replicated_param), (sharded_name, sharded_param) in zip(
        replicated_module.named_parameters(), sharded_module.named_parameters()
    ):
        clean_sharded_name = sharded_name
        for prefix in prefixes_to_ignore:
            clean_sharded_name = clean_sharded_name.replace(prefix, "")
        assert replicated_name == clean_sharded_name
        assert isinstance(sharded_param, DTensor)
        mesh, placements = sharded_param.device_mesh, sharded_param.placements
        if tuple(placements) == (Shard(0), Shard(0)):
            raise AssertionError(
                "FSDP's (Shard(0), Shard(0)) layout differs from distribute_tensor(), "
                "so we cannot check for equality using it"
            )
        sharded_ref_param = distribute_tensor(replicated_param, mesh, placements)
        assert torch.allclose(sharded_param.to_local(), sharded_ref_param.to_local())
        if replicated_param.grad is None:
            assert sharded_param.grad is None
            continue
        assert sharded_param.grad is not None
        sharded_ref_grad = distribute_tensor(replicated_param.grad, mesh, placements)
        assert isinstance(sharded_param.grad, DTensor)
        assert torch.allclose(
            sharded_param.grad.to_local(), sharded_ref_grad.to_local()
        )
