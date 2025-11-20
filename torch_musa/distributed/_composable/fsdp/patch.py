"""Patch for FSDP module"""

# pylint: disable=W0613,C0301

import os
from typing import (
    Optional,
    List,
)
from functools import wraps
import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPCommContext,
    FSDPParamGroup,
)
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    foreach_all_gather_copy_out,
    _get_param_all_gather_inputs,
    _get_all_gather_input_metadatas,
    AllGatherResult,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

__all__ = ["_apply_fsdp2_patches"]

# set this flag to False will disable the chances of overlap below:
#
# overlap allgather-copy-in with forward compute and reduce-scatter
# overlap allgather with foward/backward computation
# overlap reduce-scatter with backward computation
# overlap with all-gather/reduce-scatter/backward computation
_TORCH_MUSA_FSDP2_DISABLE_OVERLAP = False


def _init_default_fully_shard_mesh() -> DeviceMesh:
    """Default to global MUSA mesh if possible else global CPU mesh."""
    if not dist.distributed_c10d.is_initialized():
        dist.distributed_c10d.init_process_group()
    default_pg = dist.distributed_c10d._get_default_group()
    device_type = "musa" if torch.musa.is_available() else "cpu"
    mesh = init_device_mesh(device_type, mesh_shape=(default_pg.size(),))
    return mesh


def comm_context_lazy_init(self, device: torch.device):
    """set streams used by all-gather-copy-in, all-gather, reduce_scatter
    and all_reduce to current_stream.
    """
    assert device.type == torch._C._get_privateuse1_backend_name()
    if not torch.musa.is_available():
        raise RuntimeError("FSDP requires MUSA for streams")

    if _TORCH_MUSA_FSDP2_DISABLE_OVERLAP:
        current_stream = torch.musa.current_stream()
        self.all_gather_copy_in_stream = current_stream
        self.all_gather_stream = current_stream
        self.reduce_scatter_stream = current_stream
        self.all_reduce_stream = current_stream
    else:
        # Setting the all-gather/reduce-scatter streams to be higher priority
        # can help avoid some issues where their copies in/out are delayed and
        # block computation (this is different from high-pri MCCL streams)
        high_priority = -1
        # All-gather state and copy-in stream allow overlapping the next
        # copy-in with the current all-gather in forward; copy-in overlaps with
        # reduce-scatter in backward without the separate copy-in stream
        self.all_gather_copy_in_stream = torch.musa.Stream(priority=high_priority)
        # All-gather stream allows overlapping next all-gather with current
        # forward compute
        self.all_gather_stream = torch.musa.Stream(priority=high_priority)
        # Reduce-scatter stream gives separate execution "thread" for post-
        # backward logic like pre/post-gradient division and reduce-scatter
        self.reduce_scatter_stream = torch.musa.Stream(priority=high_priority)
        # Run the HSDP all-reduces concurrently with all-gather/reduce-scatter
        # since collectives use different network resources and can overlap
        # in the typical intra-node sharding / inter-node replication case
        self.all_reduce_stream = torch.musa.Stream()

    # All-gather/reduce-scatter states keep references to collective
    # tensors produced in one stream and used in another and accompanying
    # MUSA events for synchronization
    self.all_gather_state: Optional[AllGatherState] = None
    self.reduce_scatter_state: Optional[ReduceScatterState] = None
    # Post-forward order for explicit backward prefetching
    self.post_forward_order: List[FSDPParamGroup] = []  # will cause ref cycles


def unshard(self, async_op: bool = False):
    """all-gather the fsdp params"""
    if self._all_gather_result is not None:  # already called, pending wait
        return
    if self.is_unsharded:
        return  # no-op
    if self._reshard_after_forward_event is not None:
        # Resharded parameter data is allocated in the default stream and
        # used in the all-gather streams
        self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
        self._reshard_after_forward_event = None
    with record_function(self._with_fqn("FSDP::all_gather")):
        self._all_gather_result = foreach_all_gather(
            self.fsdp_params,
            self._all_gather_process_group,
            async_op,
            *self.comm_ctx.get_all_gather_streams(async_op, self._training_state),
            self.device,
        )


def wait_for_unshard(self):
    """
    1. In forward with implict prefetching, to overlap the current copy-out
    with the next all-gather, we save a reference to the current all-gather
    result to free after the next copy-out.
    2. Otherwise (explicit prefetching or in backward), we free the
    all-gather result immediately after the current copy-out since we can
    already overlap the current copy-out with the previous reduce-scatter.
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

    # [NOTE] Why save a reference of _all_gather_result in FORWARD ?
    # allgather and copy-out already use different streams
    self._all_gather_result = None  # free unless saved in `all_gather_state`


# _fsdp_collectives.py
@torch.no_grad()
def foreach_all_gather(
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: None,
    all_gather_stream: None,
    device: torch.device,
) -> Optional[AllGatherResult]:
    """foreach_all_gather

    TODO: maybe we can avoid step by step all-gather in non-overlap case
    """
    world_size, rank = group.size(), group.rank()

    # TODO: maybe we can reduce once memory allocation if no overlapping
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
    all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
        all_gather_inputs,
        inp_split_sizes,
        all_gather_input_numel,
        world_size,
        rank,
        dtype,
        device,
    )
    del param_all_gather_inputs

    all_gather_work = dist.all_gather_into_tensor(
        output_tensor=all_gather_output,
        input_tensor=all_gather_input,
        group=group,
        async_op=async_op,
    )
    all_gather_event = None  # set to None if default stream
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
    global _TORCH_MUSA_FSDP2_DISABLE_OVERLAP

    if torch.musa.core._utils._get_musa_arch() < 31:
        # disable the overlap between the computations and communications by default
        _TORCH_MUSA_FSDP2_DISABLE_OVERLAP = (
            os.environ.get("TORCH_MUSA_FSDP2_DISABLE_OVERLAP", "1") == "1"
        )
    else:
        # enable the overlap between the computations and communications by default
        _TORCH_MUSA_FSDP2_DISABLE_OVERLAP = (
            os.environ.get("TORCH_MUSA_FSDP2_DISABLE_OVERLAP", "0") == "1"
        )
    if _TORCH_MUSA_FSDP2_DISABLE_OVERLAP:
        torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_all_gather.__code__ = (
            foreach_all_gather.__code__
        )
        FSDPParamGroup.unshard = unshard
        FSDPParamGroup.wait_for_unshard = wait_for_unshard

    # we always patch this now, cause we havn't monkey-patch torch.cuda.is_available()
    # move this into scope above after PT2.6 landed
    FSDPCommContext.lazy_init = comm_context_lazy_init

    # _init_default_fully_shard_mesh is referenced by `fully_shard()`, so just replace __code__ here
    torch.distributed.fsdp._fully_shard._fsdp_init._init_default_fully_shard_mesh.__code__ = (
        _init_default_fully_shard_mesh.__code__
    )


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
    torch.cuda.Stream = torch.musa.Stream
    torch.cuda.current_device = torch.musa.current_device
    torch.cuda.current_stream = torch.musa.current_stream
    torch.cuda.set_stream = torch.musa.set_stream
    torch.cuda.Event = torch.musa.Event

    torch.distributed.fsdp.fully_shard = monkey_patched_fully_shard(
        fully_shard
    )
