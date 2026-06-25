# pylint: disable=all
""" 
adapt to lazy_hsdp_allreduce switch in post_backward
realize the all_reduce for lazy_hsdp usage
"""
import logging
from typing import Any

import torch
from torch.profiler import record_function
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    foreach_reduce,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    compiled_autograd_enabled,
    TrainingState,
    _get_dim0_padded_size
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    AllReduceState,
    ReduceScatterState,
)

from ._fsdp_collectives import (
    lazy_hsdp_all_reduce,
    lazy_hsdp_update_grads,
)

logger = logging.getLogger("torch.distributed.fsdp.fully_shard")


def post_backward(self, *unused: Any):
    # This method should be idempotent and safe to call even when this
    # FSDP parameter group was not used in backward (should be a no-op)
    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_backward"))
    self._training_state = TrainingState.POST_BACKWARD
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
        fsdp_params_with_grad: list[FSDPParam] = []
        unsharded_grads: list[torch.Tensor] = []
        for fsdp_param in self.fsdp_params:
            if not hasattr(fsdp_param, "_unsharded_param"):
                continue
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
        if (
            self.comm_ctx.reduce_scatter_state is not None
            and self.comm_ctx.reduce_scatter_state.event is not None
        ):
            self.device_handle.current_stream().wait_event(
                self.comm_ctx.reduce_scatter_state.event
            )
        self.comm_ctx.reduce_scatter_state = None
        all_reduce_pg = self._all_reduce_process_group if self._is_hsdp else None
        all_reduce_stream: torch.cuda.Stream
        if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
            # this means the native HSDP is not enabled,
            # but user may want to have a custom HSDP setup
            assert self._all_reduce_hook is not None, (
                "all reduce hook stream is specified but hook itself is missing."
            )
            all_reduce_stream = self._all_reduce_hook_stream
        else:
            all_reduce_stream = self.comm_ctx.all_reduce_stream
        if getattr(self, 'lazy_hsdp_allreduce', False):
            logger.debug("Setting up lazy HSDP all-reduce in post_backward")
            self.all_reduce_stream = all_reduce_stream
            self.fsdp_params_with_grad = fsdp_params_with_grad
            world_size = self._reduce_scatter_process_group.size()
            self.padded_unsharded_sizes = tuple(
                _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
            )
        all_reduce_grads = (
            self.all_reduce_grads
            and not getattr(self, 'lazy_hsdp_allreduce', False)
        )
        self._wait_for_post_backward()
        (
            reduce_scatter_input,
            reduce_scatter_event,
            self._post_reduce_event,
            all_reduce_input,
            all_reduce_event,
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
        self.comm_ctx.reduce_scatter_state = ReduceScatterState(
            reduce_scatter_input, reduce_scatter_event
        )
        if all_reduce_input is not None and not getattr(self, 'lazy_hsdp_allreduce', False):
            if self.device.type != "cpu":
                assert all_reduce_event is not None
            self._all_reduce_state = AllReduceState(
                all_reduce_input, all_reduce_event
            )

def post_backward_final_lazy_hsdp_all_reduce(self):
    """
    Execute the lazy HSDP all-reduce after all backwards operations are done, 
    instead of executing the all-reduce in each module's `post_backward`.
    Aim to reduce the bubble between the backwards.
    1. This function is called in the root FSDP's `_root_post_backward_final_callback`.
    2. It assumes all FSDPParamGroups with lazy HSDP all-reduce have already
        finished their reduce-scatter operations and have their
        `_partial_reduce_output` ready.
    3. It performs the all-reduce on the saved `_partial_reduce_output` and
        resets the `_partial_reduce_output` to None.
    4. It saves the all-reduce state in `_all_reduce_state` to keep the
        all-reduce input alive until the end of backward.
    """
    assert (self._partial_reduce_output is not None), (
        "HSDP all-reduce called without reduce-scatter output."
    )
    assert (len(self.fsdp_params_with_grad) != 0), (
        "HSDP all-reduce called without FSDP params with gradients."
    )

    reduce_output = self._partial_reduce_output
    
    all_reduce_group = self._all_reduce_process_group if self._is_hsdp else None
    reduce_dtype = self._reduce_dtype or self.fsdp_params_with_grad[0]._unsharded_param.dtype

    (
        all_reduce_input,
        all_reduce_event,
        self.postdivide_factor
    ) = lazy_hsdp_all_reduce(
        self._reduce_scatter_process_group,
        self.comm_ctx.reduce_scatter_stream,
        reduce_dtype,
        self.device,
        self.gradient_divide_factor,
        all_reduce_group,
        self.all_reduce_stream,
        reduce_output,
        self.force_sum_reduction_for_comms,
    )

    if all_reduce_input is not None:
        assert all_reduce_event is not None
        self._all_reduce_state = AllReduceState(
            all_reduce_input, all_reduce_event
        )

def post_backward_final_lazy_hsdp_update_grads(self):
    """
    Refresh grads for each param
    """
    self._partial_reduce_output = None

    lazy_hsdp_update_grads(
        self.fsdp_params_with_grad,
        self._reduce_scatter_process_group,
        self.comm_ctx.reduce_scatter_stream,
        self._orig_dtype,
        self.device,
        self.postdivide_factor,
        self.all_reduce_stream,
        self._all_reduce_state.all_reduce_input,
        self._all_reduce_hook,
        self.padded_unsharded_sizes,
    )


def _apply_fsdp_param_group_patch():
    (torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup
    .post_backward) = post_backward
    (torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup.
    post_backward_final_lazy_hsdp_all_reduce ) = post_backward_final_lazy_hsdp_all_reduce
    (torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup.
    post_backward_final_lazy_hsdp_update_grads ) = post_backward_final_lazy_hsdp_update_grads
