# pylint: disable=all
import os
from typing import no_type_check

import torch
import torch.distributed as dist
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _no_dispatch_record_stream,
)

from torch.distributed.fsdp._flat_param import (
    FlatParamHandle,
    HandleShardingStrategy,
)

from torch.distributed.fsdp._runtime_utils import (
    _accumulate_sharded_grad,
    _post_reduce_grad_callback,
    _get_reduce_scatter_tensors,
    _div_if_needed,
    )

# flag for compare multiple computation streams with single computation stream, default is True
# TODO(mingyuan.wang): try `_TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY=False` on musa arch'version greater than 22
_TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY = True

# flag for disable async all-reduce in HSDP, default is False
_TORCH_MUSA_FSDP_FORCE_SYNC_ALL_REDUCE = False

@no_type_check
def _init_streams(
    state: _FSDPState,
) -> None:
    """
    Initializes CUDA/CUDA-like streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    global _TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY
    assert state._is_root
    assert state._device_handle.is_available()
    uses_hybrid_sharding = any(
        fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES
        for fsdp_state in state._all_fsdp_states
    )
    default_stream_only = _TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY

    # Prioritize all-gathers/reduce-scatters over async all-reduce for HSDP and
    # preserve the default priority of 0 otherwise
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
    # Default stream for computation
    state._default_stream = state._device_handle.current_stream()
    # Stream for unshard logic, including allocating the all-gather destination
    # tensors and the all-gathers themselves
    state._unshard_stream = state._default_stream if default_stream_only else state._device_handle.Stream(priority=high_priority)
    # Stream for overlapping gradient reduction with the backward pass gradient
    # computation
    state._post_backward_stream = state._default_stream if default_stream_only else state._device_handle.Stream(priority=high_priority)
    # Stream for pre-unshard logic, namely allocations and writes for CPU
    # offloading (H2D copy) and mixed precision (low precision cast)
    state._pre_unshard_stream = state._default_stream if default_stream_only else state._device_handle.Stream(priority=high_priority)
    # Stream to run HSDP's all-reduce as async (if using HSDP)
    state._all_reduce_stream = (
        state._default_stream if default_stream_only else \
            state._device_handle.Stream() if uses_hybrid_sharding else state._default_stream
    )

@no_type_check
def _reduce_grad(state: _FSDPState, handle: FlatParamHandle) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """
    global _TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY
    flat_param = handle.flat_param
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (
        HandleShardingStrategy.HYBRID_SHARD,
        HandleShardingStrategy._HYBRID_SHARD_ZERO2,
    )
    # We clear `.grad` to permit multiple backwards. This avoids a race where
    # the second backward pass computation precedes ahead of the first backward
    # pass reduction, which is possible since the reduction is issued in a
    # separate stream and is async and would result in reducing the wrong
    # gradient.
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(
        state, unsharded_grad
    )
    if state._comm_hook is None:  # default path
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        pg = (
            handle._fake_process_group
            if handle._use_fake_reduce
            else state.process_group
        )
        dist.reduce_scatter_tensor(
            new_sharded_grad,
            padded_unsharded_grad,
            group=pg,
        )
        if uses_hybrid_sharded_strategy:
            global _TORCH_MUSA_FSDP_FORCE_SYNC_ALL_REDUCE, _TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY
            state._all_reduce_stream.wait_stream(state._post_backward_stream)
            with state._device_handle.stream(state._all_reduce_stream):
                # Since the new sharded gradient is produced in the post-
                # backward stream and consumed in the all-reduce stream,
                # inform the caching allocator
                _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
                dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                grad_to_offload = _accumulate_sharded_grad(
                    state, handle, new_sharded_grad
                )
                _post_reduce_grad_callback(state, handle, grad_to_offload)
                if state._all_reduce_stream is not state._default_stream and \
                    _TORCH_MUSA_FSDP_FORCE_SYNC_ALL_REDUCE:
                    # TODO(mingyuan.wang): implicit synchronization by not using `_all_reduce_stream` still hangs, but explicit sync works
                    state._device_handle.current_stream().synchronize()
                return
        _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(
            state._comm_hook_state, padded_unsharded_grad, new_sharded_grad
        )
        # NOTE: HSDP variants do not support communication hook.
    grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    _post_reduce_grad_callback(state, handle, grad_to_offload)

def _apply_runtime_utils_patch():
    # object level substitution should be okay
    torch.distributed.fsdp._runtime_utils._reduce_grad = (
        _reduce_grad
    )
    # torch.distributed.fsdp._runtime_utils._reduce_grad.__code__ = (
    #     _reduce_grad.__code__
    # )

    global _TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY
    global _TORCH_MUSA_FSDP_FORCE_SYNC_ALL_REDUCE

    _TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY = os.environ.get("TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY", "1") == "1"
    if _TORCH_MUSA_FSDP_DEFAULT_STREAM_ONLY:
        _TORCH_MUSA_FSDP_FORCE_SYNC_ALL_REDUCE = os.environ.get("TORCH_MUSA_FSDP_FORCE_SYNC_ALL_REDUCE", "0") == "1"

    torch.distributed.fsdp._runtime_utils._init_streams = (
        _init_streams
    )
