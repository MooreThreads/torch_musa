"""add grad accumulate function for hsdp"""
import logging

import torch
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    compiled_autograd_enabled,
    TrainingState,
)

logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

def _root_post_backward_final_callback(self) -> None:
    if not compiled_autograd_enabled():
        logger.debug("FSDP::root_post_backward")
    with torch.profiler.record_function("FSDP::root_post_backward_callback"):
        for state in self._state_ctx.all_states:
            fsdp_param_group = state._fsdp_param_group
            if (
                fsdp_param_group
                and fsdp_param_group._training_state != TrainingState.POST_BACKWARD
            ):
                fsdp_param_group.post_backward()
            state._training_state = TrainingState.IDLE
            if fsdp_param_group:
                fsdp_param_group._training_state = TrainingState.IDLE
            if self._state_ctx.is_last_backward:
                state._finalize_backward()

        if self._state_ctx.is_last_backward:
            self._comm_ctx.post_forward_order.clear()
            if self._comm_ctx.reduce_scatter_state is not None:
                self._device_handle.current_stream().wait_event(
                    self._comm_ctx.reduce_scatter_state.event
                )
                self._comm_ctx.reduce_scatter_state = None
        self._state_ctx.post_backward_final_callback_queued = False
    # Warning(by Tianyi.Tang): here cm_profiling assumes all FSDP states with lazy HSDP
    # use the same all_reduce_process_group (if they need to own one),
    # which has not been strictly guaranteed by now.
    # If groups' contents must to be different among states, just remove cm will work.
    group = None
    for state in self._state_ctx.all_states:
        if state._fsdp_param_group is not None and state._fsdp_param_group._is_hsdp:
            group = state._fsdp_param_group._all_reduce_process_group
            break
    # Additional check:
    # confirm all_reduce is only enabled for states that require this operation.
    if group is not None: #HSDP
        lazy_states = [
            state for state in self._state_ctx.all_states
            if state._fsdp_param_group
            and getattr(state._fsdp_param_group, 'lazy_hsdp_allreduce', False)
            and state._fsdp_param_group.all_reduce_grads
        ]
        with torch.distributed._coalescing_manager(group):
            for state in lazy_states:
                state._fsdp_param_group.post_backward_final_lazy_hsdp_all_reduce()

        for state in lazy_states:
            state._fsdp_param_group.post_backward_final_lazy_hsdp_update_grads()

def _apply_fsdp_state_patch():
    (torch.distributed.fsdp._fully_shard._fsdp_state.FSDPState
    ._root_post_backward_final_callback) = _root_post_backward_final_callback
