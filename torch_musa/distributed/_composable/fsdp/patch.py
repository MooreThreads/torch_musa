"""Patches for FSDP2 module"""

# pylint: disable=W0613,C0301,E1121,C0415,C0103

from typing import (
    Optional,
    List,
    Any,
)
import warnings

# from enum import Enum
from functools import wraps
import torch
import torch.distributed as dist
import torch._dynamo.compiled_autograd as ca
from torch.profiler import record_function
from torch.distributed.fsdp._fully_shard import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPParamGroup,
    logger,
)
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    foreach_all_gather_copy_out,
    _get_param_all_gather_inputs,
    _get_all_gather_input_metadatas,
    AllGatherResult,
    foreach_reduce,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState


__all__ = ["_apply_fsdp2_patches"]


# _fsdp_param_group.py
def wait_for_unshard_non_overlap(self):
    """
    the stream syncs at python side were removed in non overlap case
    """
    if not self._all_gather_result:
        return  # no preceding unshard

    with record_function(self._with_fqn("FSDP::all_gather_copy_out")):
        foreach_all_gather_copy_out(
            self._all_gather_result,
            self.fsdp_params,
            self._all_gather_process_group,
        )
    for fsdp_param in self.fsdp_params:
        fsdp_param.init_unsharded_param()
    self._to_unsharded()

    # free memory used by all-gather output
    self._all_gather_result = None  # free unless saved in `all_gather_state`


# _fsdp_param_group.py
def post_backward_non_overlap(self, *unused: Any):
    """post_backward will be used in non overlap case"""
    if not ca.compiled_autograd_enabled:
        logger.debug("%s", self._with_fqn("FSDP::post_backward"))
    self._traing_state = TrainingState.POST_BACKWARD
    with record_function(self._with_fqn("FSDP::post_backward_accumulate")):
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()
    with record_function(self._with_fqn("FSDP::post_backward_reshard")):
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.reshard()
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_accumulated_grad_if_needed()
            return
        # Save the autograd-computed gradients before resharding to only
        # access the unsharded parameters when their data is present
        fsdp_params_with_grad: List[FSDPParam] = []
        unsharded_grads: List[torch.Tensor] = []
        for fsdp_param in self.fsdp_params:
            # May have an accumulated gradient of the reduce dtype if the
            # previous backward did not reduce-scatter
            if fsdp_param.unsharded_accumulated_grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_accumulated_grad_data)
                fsdp_param.unsharded_accumulated_grad = None
            elif fsdp_param.unsharded_param.grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_grad_data)
                fsdp_param.unsharded_param.grad = None
        if self.reshard_after_backward:
            self.reshard()
    if len(fsdp_params_with_grad) == 0:
        return
    with record_function(self._with_fqn("FSDP::post_backward_reduce")):
        # See [Note: Unset reduce_scatter_state]
        # if self.comm_ctx.reduce_scatter_state is not None:
        #     torch.cuda.current_stream().wait_event(
        #         self.comm_ctx.reduce_scatter_state.event
        #     )
        #     self.comm_ctx.reduce_scatter_state = None
        (
            _,
            _,
            self._post_reduce_event,
            _,
            _,
            self._partial_reduce_output,
        ) = foreach_reduce(
            fsdp_params_with_grad,
            unsharded_grads,
            self._reduce_scatter_process_group,
            self.comm_ctx.reduce_scatter_stream,
            self._reduce_scatter_comm,
            self._orig_dtype,
            self._reduce_dtype,
            self.device,
            self.reduce_scatter_reduce_op,
            self._all_reduce_process_group if self._is_hsdp else None,
            self.comm_ctx.all_reduce_stream,
            self.all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
        )
        # [Note: Unset reduce_scatter_state]
        # the reduce-scatter input is allocated in current_stream and used in
        # reduce_scatter comm stream, but in FSDP2OverlapLevel.NO_OVERLAP case
        # its memory is safe to be reused for the later computations in current_stream,
        # so we don't need to hold reference and use MUSA events for synchronization here.
        # self.comm_ctx.reduce_scatter_state = ReduceScatterState(
        #     reduce_scatter_input, reduce_scatter_event
        # )


# _fsdp_collectives.py
@torch.no_grad()
def foreach_all_gather_non_overlap(
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: None,
    all_gather_stream: None,
    device: torch.device,
    all_gather_comm: None,
) -> Optional[AllGatherResult]:
    """
    the stream syncs at python side were removed in non overlap case, we
    are using current_stream in this case.
    """
    world_size, rank = group.size(), group.rank()

    param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
    (
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        dtype,
    ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
    if dtype == torch.uint8:
        all_gather_inputs = [
            t.view(torch.uint8) for ts in param_all_gather_inputs for t in ts
        ]
    else:
        all_gather_inputs = [t for ts in param_all_gather_inputs for t in ts]

    inp_split_sizes = [t.numel() for t in all_gather_inputs]
    all_gather_input_numel = sum(inp_split_sizes)
    all_gather_output = all_gather_comm.allocate(
        (all_gather_input_numel * world_size,), dtype=dtype, device=device
    )
    all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        all_gather_inputs,
        all_gather_output,
        inp_split_sizes,
        all_gather_input_numel,
        rank,
    )
    # safe to free, all-gather comm stream will wait copy-in finish
    del param_all_gather_inputs

    all_gather_work = all_gather_comm(
        output_tensor=all_gather_output,
        input_tensor=all_gather_input,
        group=group,
        async_op=async_op,
    )
    # all-gather copy-in/copy-out both using current_stream, just set all_gather_event to None
    all_gather_event = None
    return AllGatherResult(
        all_gather_output,
        all_gather_event,
        all_gather_work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        inp_split_sizes,
    )


# no patch on foreach_all_gather_copy_out, which running on the default/current_stream


def _setup_fsdp2_patches():
    """lazily apply patches

    Why use lazy patches here ?
    To avoid the initialization of musa state caused by _get_musa_arch during import torch_musa.
    """

    from .custom_overlap_patch import (
        _apply_custom_overlap_patch,
        FSDP2OverlapLevel,
        _FSDP2_OVERLAP_LEVEL,
    )

    _apply_custom_overlap_patch()

    if torch.musa.core._utils._get_musa_arch() < 31:
        _FSDP2_OVERLAP_LEVEL = FSDP2OverlapLevel.NO_OVERLAP
        warnings.warn(
            "The overlapping of FSDP2 was disabled on musa arch older than mp_31"
        )

    if _FSDP2_OVERLAP_LEVEL == FSDP2OverlapLevel.NO_OVERLAP:
        #
        torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_all_gather.__code__ = (
            foreach_all_gather_non_overlap.__code__
        )
        FSDPParamGroup.wait_for_unshard = wait_for_unshard_non_overlap
        FSDPParamGroup.post_backward = post_backward_non_overlap


def monkey_patched_fully_shard(fully_shard_func):
    """Returns monkey patched fully_shard function, which will apply our patches on the first invocation"""
    has_patched = False

    @wraps(fully_shard_func)
    def wrapper(*args, **kwargs):
        nonlocal has_patched
        if not has_patched:
            _setup_fsdp2_patches()
            has_patched = True
        return fully_shard_func(*args, **kwargs)

    return wrapper


def _apply_fsdp2_patches():
    torch.distributed.fsdp.fully_shard = monkey_patched_fully_shard(fully_shard)
