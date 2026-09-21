"""Patches for FSDP2 module"""

# pylint: disable=W0613,C0301,E1121,C0415,C0103,W0611

from typing import (
    Optional,
    List,
    Any,
)
import warnings
from itertools import chain

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
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, alloc_storage
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    TrainingState,
    _get_dim0_padded_size,
)


__all__ = ["_apply_fsdp2_patches"]


# _fsdp_param_group.py
def wait_for_unshard_non_overlap(self):
    """
    the stream syncs at python side were removed in non overlap case
    """
    if not self._all_gather_result:
        return  # no preceding unshard

    world_size = self._all_gather_process_group.size()

    if world_size == 1:
        # directly initialize unsharded parameters from sharded parameters
        for fsdp_param in self.fsdp_params:
            # Use all_gather_inputs which already handles conversion to param_dtype
            # This is consistent with the world_size > 1 path
            all_gather_input = fsdp_param.all_gather_inputs[0]

            # Make sure the all_gather_outputs has proper storage size before using it
            # First ensure we have at least one tensor in all_gather_outputs
            fsdp_param.init_all_gather_outputs(
                [all_gather_input.numel()],
                [all_gather_input.dtype],
                world_size,
                self.device,
                force_recreate=False,
            )

            tensor = fsdp_param.all_gather_outputs[0]
            alloc_storage(tensor)

            # find alternative way to check if tensor.is_inference
            with torch.autograd._unsafe_preserve_version_counter(tensor):
                tensor.copy_(all_gather_input)
    else:
        with record_function(self._with_fqn("FSDP::all_gather_copy_out")):
            foreach_all_gather_copy_out(
                self._all_gather_result,
                self.fsdp_params,
                self._all_gather_process_group,
            )
    for fsdp_param in self.fsdp_params:
        fsdp_param.init_unsharded_param()
    self._to_unsharded()

    # in non overlap case, we don't need to defer free `_all_gather_result` or
    # let all_gather stream waits previous all_gather_copy_out done explicitly.

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
        all_reduce_pg = self._all_reduce_process_group if self._is_hsdp else None
        all_reduce_stream: torch.musa.Stream
        if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
            # this means the native HSDP is not enabled,
            # but user may want to have a custom HSDP setup
            assert (
                self._all_reduce_hook is not None
            ), "all reduce hook stream is specified but hook itself is missing."
            all_reduce_stream = self._all_reduce_hook_stream
        else:
            all_reduce_stream = self.comm_ctx.all_reduce_stream
        lazy_hsdp_allreduce = getattr(self, "lazy_hsdp_allreduce", False)
        if lazy_hsdp_allreduce:
            logger.debug("Setting up lazy HSDP all-reduce in post_backward")
            self.all_reduce_stream = all_reduce_stream
            self.fsdp_params_with_grad = fsdp_params_with_grad
            world_size = self._reduce_scatter_process_group.size()
            self.padded_unsharded_sizes = tuple(
                _get_dim0_padded_size(grad.size(), world_size)
                for grad in unsharded_grads
            )
        all_reduce_grads = self.all_reduce_grads and not lazy_hsdp_allreduce
        self._wait_for_post_backward()
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
            self.gradient_divide_factor,
            self._all_reduce_process_group if self._is_hsdp else None,
            all_reduce_stream,
            all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
            self.force_sum_reduction_for_comms,
        )
        # [Note: Unset reduce_scatter_state]
        # the reduce-scatter input is allocated in current_stream and used in
        # reduce_scatter comm stream, but in FSDP2OverlapLevel.NO_OVERLAP case
        # its memory is safe to be reused for the later computations in current_stream,
        # so we don't need to hold reference and use MUSA events for synchronization here.

        # [Note: Unset all_reduce_state]
        # when lazy HSDP all-reduce is disabled, all-reduce and the later
        # `_to_dtype_if_needed(reduce_output, orig_dtype)` run in current_stream,
        # so the all-reduce input does not need an extra reference to extend
        # its lifetime across streams; when lazy HSDP all-reduce is enabled,
        # `foreach_reduce()` saves the reduce-scatter output in
        # `_partial_reduce_output` for the root final callback instead.


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
        all_gather_inputs = [*chain.from_iterable(param_all_gather_inputs)]

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
    # TODO(mingyuan.wang): Drop these NO_OVERLAP patches once the overlap path is stable enough.
    if _FSDP2_OVERLAP_LEVEL == FSDP2OverlapLevel.NO_OVERLAP:
        torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_all_gather.__code__ = (
            foreach_all_gather_non_overlap.__code__
        )
        FSDPParamGroup.wait_for_unshard = wait_for_unshard_non_overlap
        FSDPParamGroup.post_backward = post_backward_non_overlap


def monkey_patched_fully_shard(fully_shard_func):
    """Returns monkey patched fully_shard function, which will apply our patches on the first invocation"""
    has_patched = False
    from .custom_overlap_patch import _maybe_set_custom_comm

    @wraps(fully_shard_func)
    def wrapper(*args, **kwargs):
        nonlocal has_patched
        if not has_patched:
            _setup_fsdp2_patches()
            has_patched = True
        fsdp_module = fully_shard_func(*args, **kwargs)
        _maybe_set_custom_comm(fsdp_module)
        return fsdp_module

    return wrapper


def _apply_fsdp2_patches():
    torch.distributed.fsdp.fully_shard = monkey_patched_fully_shard(fully_shard)
