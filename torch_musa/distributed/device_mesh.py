# pylint: disable=missing-module-docstring, line-too-long
from typing import List, Optional
import torch
from torch.distributed.device_mesh import (
    DeviceMesh,
    _get_default_group,
    is_initialized,
    init_process_group,
    get_rank,
    get_world_size,
    _get_device_handle,
)

# If backend is None, both gloo and nccl backend will be created by `init_process_group`,
# so here we might need to specify the backend explicitly.
# MCCL was registered as a new backend after _MUSAC was imported, then we could
# get this MCCL backend from device_backend_map.
def _get_or_create_default_group(self):
    default_initialized = is_initialized()
    if not default_initialized:
        init_process_group(
            backend=torch.distributed.Backend.default_device_backend_map.get(
                self.device_type.lower(), None
            )
        )

    world_size = get_world_size()
    if self.mesh.numel() > world_size:
        raise RuntimeError(
            f"Mesh should not be bigger than default world size, but found {self.mesh.numel()} ranks!"
        )

    device_handle = _get_device_handle(self.device_type)
    # TODO: if user want to pass pg_options, offer a way to do it
    if not default_initialized and device_handle:
        # automatically set the current cuda/cuda-like device base on num of gpu devices available in each host
        # NOTE: This device selection would only work for homogeneous hardware.
        num_devices_per_host = device_handle.device_count()
        if world_size > num_devices_per_host and world_size % num_devices_per_host != 0:
            raise RuntimeError(
                f"DeviceMesh only support homogeneous hardware, but found "
                f"{world_size} ranks and {num_devices_per_host} {self.device_type} devices!"
            )
        device_handle.set_device(get_rank() % num_devices_per_host)

    # calculate the coordinates of the current global rank on the mesh
    rank_coords = (self.mesh == get_rank()).nonzero()
    assert rank_coords.size(0) in (0, 1)
    self._coordinate_on_dim: Optional[List[int]] = (
        rank_coords[0].tolist() if rank_coords.size(0) > 0 else None
    )
    return _get_default_group()


def _apply_device_mesh_patch():
    DeviceMesh._get_or_create_default_group = (
        _get_or_create_default_group
    )
