"""Custom FSDP2 overlap settings"""

# pylint: disable=C0301,C0415,W0602,C0103,R1710
import logging
import os
import warnings
from enum import Enum
from collections.abc import Sequence
from typing import (
    List,
    Optional,
    Union,
    Any,
)
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPParamGroup,
    AllGatherState,
    ReduceScatterState,
)
from torch.distributed.fsdp._fully_shard._fsdp_state import (
    FSDPCommContext,
)
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    AllGather,
    ReduceScatter,
    _ReduceOp,
)

import torch.distributed._symmetric_memory as symm_mem
from torch.profiler import record_function


logger = logging.getLogger(__name__)


class IntraNodeLowContentionCommAllocMixin:
    """Allocates FSDP2 communication buffers from the low-contention symmetric mempool."""

    # use global singleton memory pool for all-gather and reduce-scatter to
    # maximize the memory multiplexing.
    _mem_pool = None

    def __init__(self, group: dist.ProcessGroup, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if IntraNodeLowContentionCommAllocMixin._mem_pool is None:
            device = torch.musa.current_device()
            allocator = symm_mem.get_mempool_allocator(device)
            IntraNodeLowContentionCommAllocMixin._mem_pool = torch.musa.MemPool(
                allocator
            )

        self._group = group
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(dist._get_process_group_name(group))

    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        with torch.musa.use_mem_pool(IntraNodeLowContentionCommAllocMixin._mem_pool):
            return torch.empty(*size, dtype=dtype, device=device)


class IntraNodeLowContentionCommAllGather(
    IntraNodeLowContentionCommAllocMixin, AllGather
):
    """Runs FSDP2 all-gather via the low-contention symmetric-memory kernel."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__(group)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        assert not async_op, "async all-gather is not supported currently"
        group_name = torch.distributed._get_process_group_name(group)

        with record_function("low_contention_all_gather"):
            torch.ops.symm_mem.low_contention_all_gather(
                output_tensor,
                input_tensor,
                group_name,
            )


class IntraNodeLowContentionCommReduceScatter(
    IntraNodeLowContentionCommAllocMixin, ReduceScatter
):
    """Runs FSDP2 reduce-scatter via the low-contention symmetric-memory kernel."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__(group)

        self._op_to_str = {ReduceOp.SUM: "sum", ReduceOp.AVG: "avg"}
        # FSDP2 allocates the reduce-scatter input before its output.  The
        # low-contention kernel allows the input/send buffer to be native
        # memory, while the output/receive buffer can use symmetric memory.
        self._fsdp2_allow_native_buffer_allocation = True

    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self._fsdp2_allow_native_buffer_allocation:
            # Use native memory for the first allocation in each
            # reduce-scatter call (the input/send buffer).
            self._fsdp2_allow_native_buffer_allocation = False
            return torch.empty(*size, dtype=dtype, device=device)

        # Subsequent communication buffers, including the output/receive
        # buffer, must come from the symmetric mempool used by the kernel.
        with torch.musa.use_mem_pool(IntraNodeLowContentionCommAllocMixin._mem_pool):
            return torch.empty(*size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        # Prepare the allocator for the next FSDP2 reduce-scatter cycle.  The
        # two allocate() calls happen before this collective is invoked.
        self._fsdp2_allow_native_buffer_allocation = True
        assert not async_op, "async all-gather is not supported currently"
        group_name = torch.distributed._get_process_group_name(group)

        reduce_str = self._op_to_str.get(op, None)
        if not reduce_str:
            raise RuntimeError(f"Unsupported reduce op: {op}")

        with record_function("low_contention_reduce_scatter"):
            torch.ops.symm_mem.low_contention_reduce_scatter(
                output_tensor,
                input_tensor,
                reduce_str,
                group_name,
            )


class ProcessGroupMCCLSymmMemAllocMixin:
    """Allocates FSDP2 communication buffers from a ProcessGroup MCCL symmetric mempool."""

    _mem_pool: torch.musa.MemPool = None

    def __init__(self, group: dist.ProcessGroup, *args: Any, **kwargs: Any):
        self._group = group
        super().__init__(*args, **kwargs)
        if ProcessGroupMCCLSymmMemAllocMixin._mem_pool is None:
            dist.barrier(group)  # ensure communicator is initialized
            device = torch.device(torch.musa.current_device())
            backend = group._get_backend(device)
            ProcessGroupMCCLSymmMemAllocMixin._mem_pool = torch.musa.MemPool(
                backend.mem_allocator
            )
            backend.register_mem_pool(
                ProcessGroupMCCLSymmMemAllocMixin._mem_pool, symm=True
            )

    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        with torch.musa.use_mem_pool(ProcessGroupMCCLSymmMemAllocMixin._mem_pool):
            return torch.empty(*size, dtype=dtype, device=device)


class ProcessGroupMCCLSymmMemAllGather(ProcessGroupMCCLSymmMemAllocMixin, AllGather):
    """Runs FSDP2 all-gather with MCCL using symmetric-mempool-allocated buffers."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__(group)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        return dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


class ProcessGroupMCCLSymmMemReduceScatter(
    ProcessGroupMCCLSymmMemAllocMixin, ReduceScatter
):
    """Runs FSDP2 reduce-scatter with MCCL using symmetric-mempool-allocated buffers."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__(group)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work:
        return dist.reduce_scatter_tensor(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


_TORCH_MUSA_FSDP2_COMM_TYPE_ENV = "TORCH_MUSA_FSDP2_COMM_TYPE"


def _maybe_set_custom_comm(fsdp_module: FSDPModule | list[FSDPModule]) -> None:
    """Install MUSA-specific FSDP2 collective implementations when requested.

    This replaces the module's FSDP parameter-group all-gather and
    reduce-scatter communication objects according to
    ``TORCH_MUSA_FSDP2_COMM_TYPE``. A value of 0 keeps the default FSDP2
    collectives. Values 1 and 2 enable custom intra-node communication paths
    intended to improve computation/communication overlap:

    * 1 uses low-contention symmetric-memory all-gather/reduce-scatter ops.
    * 2 uses MCCL collectives with buffers allocated from a symmetric mempool.

    Custom communication is only enabled on MUSA arch >= mp_31. In common
    FSDP2 overlap workloads, type 1 generally has better memory consumption
    than type 2.
    """

    # Multi-module groups share one FSDP state, so the first module represents
    # the group's parameter group.
    if isinstance(fsdp_module, list):
        if not fsdp_module:
            return
        fsdp_module = fsdp_module[0]

    fsdp_param_group = fsdp_module._get_fsdp_state()._fsdp_param_group

    arch_version = torch.musa.core._utils._get_musa_arch()
    # we disable custom all-gather/reduce-scatter by default
    custom_comm_type = (
        int(os.environ.get(_TORCH_MUSA_FSDP2_COMM_TYPE_ENV, 0))
        if arch_version >= 31
        else 0
    )

    if custom_comm_type == 0:
        # no computation-communicator overlap
        pass
    elif custom_comm_type == 1:
        fsdp_param_group._all_gather_comm = IntraNodeLowContentionCommAllGather(
            fsdp_param_group._all_gather_process_group
        )
        fsdp_param_group._reduce_scatter_comm = IntraNodeLowContentionCommReduceScatter(
            fsdp_param_group._reduce_scatter_process_group
        )
    elif custom_comm_type == 2:
        fsdp_param_group._all_gather_comm = ProcessGroupMCCLSymmMemAllGather(
            fsdp_param_group._all_gather_process_group
        )
        fsdp_param_group._reduce_scatter_comm = ProcessGroupMCCLSymmMemReduceScatter(
            fsdp_param_group._reduce_scatter_process_group
        )
    else:
        raise RuntimeError(f"Invalid option: {custom_comm_type}")


__all__ = ["_apply_custom_overlap_patch", "FSDP2OverlapLevel", "_FSDP2_OVERLAP_LEVEL"]


# The FSDP2 of PyTorch's implementation has the overlappings below:
# overlap allgather-copy-in with forward compute and reduce-scatter
# overlap allgather with foward/backward computation
# overlap reduce-scatter with backward computation
# overlap with all-gather/reduce-scatter/backward computation

# The overlapping schema when using different FSDP2OverlapLevel:
# CPYIN: copy-in; AG: all-gather; CPYOUT: copy-out
# 0: default stream: CPYIN0    CPYOUT0    COM0    CPYIN1    CPYOUT1    COM1
#    comm stream   :       AG0                          AG1
#
# 1: default stream: CPYIN0    CPYOUT0   CPYIN1    COM0    CPYOUT1    CPYIN2    COM1
#    comm stream   :       AG0                     AG1                          AG2
#
# 2: default stream:            CPYOUT0     COM0     CPYOUT1     COM1
#    copy-in stream: CPYIN0     CPYIN1(or later)     CPYIN2(or later)
#    comm stream   :        AG0             AG1                  AG2
#
# 3: default stream:               CPYOUT0  COM0    CPYOUT1    COM1
#    copy-in stream: CPYIN0 CPYIN1
#    comm stream   :         AG0    AG1


class FSDP2OverlapLevel(Enum):
    """FSDP2OverlapLevel enums"""

    # no overlap between communication and computation,
    # this is our default setting currently.
    NO_OVERLAP = 0

    # enable the next layer's all-gather to be overlapped with previous layer's computation
    # in this setting, overlap ONLY WORKS using explicit prefetch
    # maybe even more performant compared to OVERLAP_FSDP_COMM_COPY_IN_WITH_COMM ?
    OVERLAP_FSDP_COMM_ONLY = 1

    # enable the next layer's copy-in to be overlapped with previous layer's copy-out
    OVERLAP_FSDP_COMM_COPY_IN_WITH_COPY_OUT = 2

    # enable the next layer's copy-in to be overlapped with previous layer's all-gather
    OVERLAP_FSDP_COMM_COPY_IN_WITH_COMM = 3

    # enable the previous layer's all-reduce of grads to be overlapped with next layer's
    # computation, all-gather and reduce-scatter, which is same as PyTorch's implementation.
    OVERLAP_HSDP_COMM = 4

    END = 5


def _get_fsdp2_overlap_level() -> FSDP2OverlapLevel:
    """Resolve the FSDP2 computation/communication overlap policy.

    ``TORCH_MUSA_FSDP2_OVERLAP_LEVEL`` has the highest priority and maps
    directly to ``FSDP2OverlapLevel`` values 0 through 4. Most users do not
    need to set it unless explicitly selecting or debugging an overlap mode.
    If it is unset, keep overlap disabled by default, except for the deprecated
    ``TORCH_MUSA_FSDP2_DISABLE_OVERLAP=0`` opt-in or when custom FSDP2
    communication is enabled on MUSA arch >= mp_31.
    """
    if "TORCH_MUSA_FSDP2_OVERLAP_LEVEL" in os.environ:
        overlap_level = int(os.environ["TORCH_MUSA_FSDP2_OVERLAP_LEVEL"])
        assert overlap_level < FSDP2OverlapLevel.END.value
        if overlap_level == 0:
            return FSDP2OverlapLevel.NO_OVERLAP
        if overlap_level == 1:
            warnings.warn(
                "users should set prefetch order manually if "
                "using FSDP2OverlapLevel.OVERLAP_FSDP_COMM_ONLY"
            )
            return FSDP2OverlapLevel.OVERLAP_FSDP_COMM_ONLY
        if overlap_level == 2:
            return FSDP2OverlapLevel.OVERLAP_FSDP_COMM_COPY_IN_WITH_COPY_OUT
        if overlap_level == 3:
            return FSDP2OverlapLevel.OVERLAP_FSDP_COMM_COPY_IN_WITH_COMM
        return FSDP2OverlapLevel.OVERLAP_HSDP_COMM

    if os.environ.get("TORCH_MUSA_FSDP2_DISABLE_OVERLAP", "1") == "0":
        warnings.warn(
            "TORCH_MUSA_FSDP2_DISABLE_OVERLAP env will be deprecated in the future, use TORCH_MUSA_FSDP2_OVERLAP_LEVEL instead"
        )
        return FSDP2OverlapLevel.OVERLAP_HSDP_COMM

    if (
        os.environ.get(_TORCH_MUSA_FSDP2_COMM_TYPE_ENV, "0") != "0"
        and torch.musa.core._utils._get_musa_arch() >= 31
    ):
        # this OverlapLevel should be efficient enough
        return FSDP2OverlapLevel.OVERLAP_FSDP_COMM_COPY_IN_WITH_COPY_OUT

    return FSDP2OverlapLevel.NO_OVERLAP


_FSDP2_OVERLAP_LEVEL = _get_fsdp2_overlap_level()


def comm_context_lazy_init(self, device: torch.device):
    """setup streams will be used for communication and computation according
    to the different FSDP2 overlap strategy
    """
    self.device_handle = _get_device_handle(device.type)

    # pylint: disable=W0602
    global _FSDP2_OVERLAP_LEVEL

    current_stream = torch.musa.current_stream()

    # Setting the all-gather/reduce-scatter streams to be higher priority
    # can help avoid some issues where their copies in/out are delayed and
    # block computation (this is different from high-pri MCCL streams)
    high_priority = -1

    if _FSDP2_OVERLAP_LEVEL == FSDP2OverlapLevel.NO_OVERLAP:
        self.all_gather_copy_in_stream = current_stream
        self.all_gather_stream = current_stream
        self.reduce_scatter_stream = current_stream
        self.all_reduce_stream = current_stream
    elif _FSDP2_OVERLAP_LEVEL == FSDP2OverlapLevel.OVERLAP_FSDP_COMM_ONLY:
        self.all_gather_copy_in_stream = current_stream
        self.all_gather_stream = torch.musa.Stream(priority=high_priority)

        # we keep the same semantic with NCCL/MCCL for the executation order
        # of all_gather and reduce_scatter, i.e., all_gather and reduce_scatter
        # runs synchronous.
        self.reduce_scatter_stream = self.all_gather_stream
        self.all_reduce_stream = current_stream
    elif (
        _FSDP2_OVERLAP_LEVEL
        == FSDP2OverlapLevel.OVERLAP_FSDP_COMM_COPY_IN_WITH_COPY_OUT
    ):
        self.all_gather_copy_in_stream = torch.musa.Stream(priority=high_priority)
        self.all_gather_stream = self.all_gather_copy_in_stream
        self.reduce_scatter_stream = self.all_gather_stream
        self.all_reduce_stream = current_stream
    elif _FSDP2_OVERLAP_LEVEL == FSDP2OverlapLevel.OVERLAP_FSDP_COMM_COPY_IN_WITH_COMM:
        self.all_gather_copy_in_stream = torch.musa.Stream(priority=high_priority)
        self.all_gather_stream = torch.musa.Stream(priority=high_priority)
        self.reduce_scatter_stream = self.all_gather_stream
        self.all_reduce_stream = current_stream
    elif _FSDP2_OVERLAP_LEVEL == FSDP2OverlapLevel.OVERLAP_HSDP_COMM:
        # This is the default settings of PyTorch FSDP2
        self.all_gather_copy_in_stream = torch.musa.Stream(priority=high_priority)
        self.all_gather_stream = torch.musa.Stream(priority=high_priority)
        self.reduce_scatter_stream = torch.musa.Stream(priority=high_priority)
        self.all_reduce_stream = torch.musa.Stream()
    else:
        raise RuntimeError(f"Unexcepted fsdp2_overlap_level: {_FSDP2_OVERLAP_LEVEL}")

    # All-gather/reduce-scatter states keep references to collective
    # tensors produced in one stream and used in another and accompanying
    # MUSA events for synchronization
    self.all_gather_state: Optional[AllGatherState] = None
    self.reduce_scatter_state: Optional[ReduceScatterState] = None
    # Post-forward order for explicit backward prefetching
    self.post_forward_order: List[FSDPParamGroup] = []  # will cause ref cycles


def _apply_custom_overlap_patch():
    FSDPCommContext.lazy_init = comm_context_lazy_init
