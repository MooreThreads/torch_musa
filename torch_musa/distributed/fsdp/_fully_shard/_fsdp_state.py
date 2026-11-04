"""add grad accumulate function for hsdp"""
import logging
from collections import defaultdict

import torch
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    TrainingState,
)

logger = logging.getLogger("torch.distributed.fsdp.fully_shard")


def _root_post_backward_final_callback(self) -> None:
    with torch.profiler.record_function("FSDP::root_post_backward_callback"):
        self._state_ctx.iter_forward_root = None
        lazy_groups_by_pg = defaultdict(list)
        for state in self._state_ctx.all_states:
            state._modules_to_run_forward.clear()
            for fsdp_param_group in reversed(state._fsdp_param_groups):
                if fsdp_param_group._training_state != TrainingState.POST_BACKWARD:
                    fsdp_param_group.post_backward()
                if getattr(fsdp_param_group, "lazy_hsdp_allreduce", False) and \
                    fsdp_param_group.all_reduce_grads and \
                    fsdp_param_group._is_hsdp and \
                    fsdp_param_group._partial_reduce_output is not None:

                    process_group = fsdp_param_group._all_reduce_process_group
                    if process_group is None:
                        raise AssertionError(
                            "Lazy HSDP all-reduce requires an all-reduce process group"
                        )
                    lazy_groups_by_pg[process_group].append(fsdp_param_group)
                fsdp_param_group._training_state = TrainingState.IDLE
            state._training_state = TrainingState.IDLE

        for process_group, groups in lazy_groups_by_pg.items():
            with torch.distributed._coalescing_manager(process_group):
                for fsdp_param_group in groups:
                    fsdp_param_group.post_backward_final_lazy_hsdp_all_reduce()
            for fsdp_param_group in groups:
                fsdp_param_group.post_backward_final_lazy_hsdp_update_grads()

        if self._state_ctx.is_last_backward:
            for state in self._state_ctx.all_states:
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.finalize_backward()
            self._comm_ctx.post_forward_order.clear()
            for rs_state in self._comm_ctx.reduce_scatter_states:
                if rs_state.event is not None:
                    self._device_handle.current_stream().wait_event(
                        rs_state.event
                    )
            self._comm_ctx.reduce_scatter_states.clear()
        self._state_ctx.post_backward_final_callback_queued = False

def _apply_fsdp_state_patch():
    (torch.distributed.fsdp._fully_shard._fsdp_state.FSDPState
    ._root_post_backward_final_callback) = _root_post_backward_final_callback
