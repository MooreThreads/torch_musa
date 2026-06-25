"""add collectives for lazy_hsdp_allreduce"""
from typing import Callable, Tuple, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _get_device_handle,
    _get_gradient_divide_factors,
    _div_if_needed,
    _to_dtype_if_needed,
)

@torch.no_grad()
def lazy_hsdp_all_reduce(
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    gradient_divide_factor: Optional[float],
    all_reduce_group: dist.ProcessGroup,
    all_reduce_stream: torch.Stream,
    reduce_output: torch.Tensor,
    force_sum_reduction_for_comms: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Event,
    ]:
    """execute all_reduce"""
    (_, postdivide_factor, _, all_reduce_op) = (
        _get_gradient_divide_factors(
            reduce_scatter_group,
            all_reduce_group,
            reduce_dtype,
            device.type,
            gradient_divide_factor,
            force_sum_reduction_for_comms,
        )
    )

    all_reduce_stream.wait_stream(reduce_scatter_stream)
    device_handle = _get_device_handle(device.type)

    with device_handle.stream(all_reduce_stream):
        dist.all_reduce(
            reduce_output,
            group=all_reduce_group,
            op=all_reduce_op,
        )
        all_reduce_event = all_reduce_stream.record_event()

    # -- END: ops in reduce_scatter stream
    return (
        reduce_output,
        all_reduce_event,
        postdivide_factor,
    )

def lazy_hsdp_update_grads(
    fsdp_params: list[FSDPParam],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    orig_dtype: torch.dtype,
    device: torch.device,
    postdivide_factor: Optional[float],
    all_reduce_stream: torch.Stream,
    reduce_output: torch.Tensor,
    all_reduce_hook: Callable[[torch.Tensor], None],
    padded_unsharded_sizes: Tuple[torch.Size, ...],
    ):
    """update grads with all_reduce result"""
    world_size = reduce_scatter_group.size()
    device_handle = _get_device_handle(device.type)
    post_reduce_stream = all_reduce_stream
    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) if not
                # accumulating gradients since the CPU add kernel depends on
                # the copy result and we cannot run the add as a callback
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
            if to_accumulate_grad:
                assert isinstance(fsdp_param.sharded_param.grad, DTensor)
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
                    new_sharded_grad
                )
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            if not compiled_autograd_enabled():
                for hook in (
                    getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {})
                    or {}
                ).values():
                    hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel


def _apply_fsdp_collectives_patch():
    (torch.distributed.fsdp._fully_shard
     ._fsdp_collectives.lazy_hsdp_all_reduce) = lazy_hsdp_all_reduce
    (torch.distributed.fsdp._fully_shard
     ._fsdp_collectives.lazy_hsdp_update_grads) = lazy_hsdp_update_grads
