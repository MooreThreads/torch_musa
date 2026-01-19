"""Custom FSDP2 overlap settings"""

# pylint: disable=C0301,C0415,W0602,C0103
import logging
import os
import warnings
import functools
from enum import Enum
from collections.abc import Sequence
from typing import (
    List,
    Optional,
    Union,
    Tuple,
    Any,
    Dict,
)
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPParamGroup,
    AllGatherState,
    ReduceScatterState,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_state import (
    FSDPState,
    FSDPCommContext,
    disable_if_config_true,
    tree_map,
    _cast_fp_tensor,
)
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    AllGather,
    ReduceScatter,
    _ReduceOp,
)
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    DefaultAllGather,
    DefaultReduceScatter,
)

logger = logging.getLogger(__name__)

try:
    import ce_comm

    _ce_comm_is_available = True
except ImportError:
    _ce_comm_is_available = False

if _ce_comm_is_available:

    class CECommAllocMixin:
        """Define how to allocate tensor when using CEComm"""

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)

        def allocate(
            self,
            size: Sequence[Union[int, torch.SymInt]],
            *,
            dtype: torch.dtype,
            device: torch.device,
        ) -> torch.Tensor:
            return torch.empty(*size, dtype=dtype, device=device)

    class CECommAllGather(CECommAllocMixin, AllGather):
        """Wrapper of ce_comm all-gather"""

        def __init__(self) -> None:
            super().__init__()
            self._all_gather_into_tensor = ce_comm.all_gather_into_tensor

        def __call__(
            self,
            output_tensor: torch.Tensor,
            input_tensor: torch.Tensor,
            group: dist.ProcessGroup,
            async_op: bool = False,
        ) -> Optional[dist.Work]:
            return self._all_gather_into_tensor(
                output_tensor,
                input_tensor,
                group=group,
                async_op=async_op,
            )

    class CECommReduceScatter(CECommAllocMixin, ReduceScatter):
        """Wrapper of ce_comm reduce-scatter"""

        def __init__(self) -> None:
            super().__init__()
            self._reduce_scatter_tensor = ce_comm.reduce_scatter_tensor

        def __call__(
            self,
            output_tensor: torch.Tensor,
            input_tensor: torch.Tensor,
            group: dist.ProcessGroup,
            op: _ReduceOp,
            async_op: bool = False,
        ) -> Optional[dist.Work]:
            return self._reduce_scatter_tensor(
                output=output_tensor,
                input=input_tensor,
                group=group,
                op=op,
                async_op=async_op,
            )

else:
    CECommAllGather = None
    CECommReduceScatter = None


_TORCH_MUSA_FSDP2_ENABLE_CE_COMM_ENV = os.environ.get(
    "TORCH_MUSA_FSDP2_ENABLE_CE_COMM", None
)
_TORCH_MUSA_FSDP2_ENABLE_CE_COMM = 0
_forward_all_gather_comm = DefaultAllGather()
_backward_all_gather_comm = DefaultAllGather()
_reduce_scatter_comm = DefaultReduceScatter()


if _TORCH_MUSA_FSDP2_ENABLE_CE_COMM_ENV is not None:
    assert _TORCH_MUSA_FSDP2_ENABLE_CE_COMM_ENV in ["0", "1"]
    if _TORCH_MUSA_FSDP2_ENABLE_CE_COMM_ENV == "1":
        assert _ce_comm_is_available
        _TORCH_MUSA_FSDP2_ENABLE_CE_COMM = 1
        _forward_all_gather_comm = CECommAllGather()
        _backward_all_gather_comm = CECommAllGather()
        _reduce_scatter_comm = CECommReduceScatter()
        logger.info(
            "Using intra-node all-gather/reduce-scatter overlapped implementation"
        )
else:
    # route to overlap comms if ce_comm is available
    if _ce_comm_is_available:
        _TORCH_MUSA_FSDP2_ENABLE_CE_COMM = 1
        _forward_all_gather_comm = CECommAllGather()
        _backward_all_gather_comm = CECommAllGather()
        _reduce_scatter_comm = CECommReduceScatter()
        logger.info(
            "ce_comm is available, using intra-node all-gather/reduce-scatter overlapped implementation by default"
        )


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
    """Setup FSDP2OverlapLevel according to the TORCH_MUSA_FSDP2_OVERLAP_LEVEL env,
    if env is not specified, use FSDP2OverlapLevel.NO_OVERLAP by default.
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

    if _ce_comm_is_available:
        # this OverlapLevel should be efficient enough
        return FSDP2OverlapLevel.OVERLAP_FSDP_COMM_COPY_IN_WITH_COPY_OUT

    return FSDP2OverlapLevel.NO_OVERLAP


_FSDP2_OVERLAP_LEVEL = _get_fsdp2_overlap_level()


def comm_context_lazy_init(self):
    """setup streams will be used for communication and computation according
    to the different FSDP2 overlap strategy
    """
    if not torch.musa.is_available():
        raise RuntimeError("FSDP requires MUSA for streams")

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

    # How many fsdp layers' collective operations have been set by pre_forward hook,
    # which indicates should we stop to use mccl's collectives or not.
    # Currently, layers that cannot be overlapped still need to use mccl for better performance,
    # In principle, only the first layer or the previous few layers' communications
    # will not be overlapped.
    self._num_layer_collective_sets: int = 0


# experimental setting for transformer-like models
_MAYBE_NONOVERLAPPED_FSDP_LAYER_NUM = 2


# TODO(mingyuan.wang): delete this once zero copy all-gather/reduce-scatter over ACE implementation ready
def _maybe_set_mccl_collectives(state: FSDPState):
    """for layers that won't be overlapped, still use mccl's all-gather/reduce-scatter

    This should be called by the pre_forward hook, since the order can only be determined
    when the model's forward is actually executed.
    """
    # pylint: disable=global-variable-not-assigned
    global _FSDP2_OVERLAP_LEVEL, _TORCH_MUSA_FSDP2_ENABLE_CE_COMM

    if _TORCH_MUSA_FSDP2_ENABLE_CE_COMM and state._fsdp_param_group:
        if (
            state._comm_ctx._num_layer_collective_sets
            < _MAYBE_NONOVERLAPPED_FSDP_LAYER_NUM
        ):
            state._fsdp_param_group._forward_all_gather_comm = DefaultAllGather()
            state._fsdp_param_group._reduce_scatter_comm = DefaultReduceScatter()
            # shared for all FSDPStates
            state._comm_ctx._num_layer_collective_sets += 1


def _fsdp_state_init_shared_state(self) -> None:
    global _FSDP2_OVERLAP_LEVEL
    self._comm_ctx.lazy_init()
    for state in self._state_ctx.all_states:
        state._state_ctx = self._state_ctx
        state._comm_ctx = self._comm_ctx
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.comm_ctx = self._comm_ctx

            # register all-gather/reduce-scatter for forward/backward prop
            if _FSDP2_OVERLAP_LEVEL != FSDP2OverlapLevel.NO_OVERLAP:
                fsdp_param_group._forward_all_gather_comm = _forward_all_gather_comm
                fsdp_param_group._backward_all_gather_comm = _backward_all_gather_comm
                fsdp_param_group._reduce_scatter_comm = _reduce_scatter_comm


@disable_if_config_true
def _pre_forward(
    self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    # When composing with module-hook-based activation checkpointing, the
    # the pre-backward hook is responsible for the unshard
    if self._training_state == TrainingState.PRE_BACKWARD:
        return args, kwargs
    self._training_state = TrainingState.FORWARD
    args, kwargs = self._root_pre_forward(module, args, kwargs)
    if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
        with torch.profiler.record_function("FSDP::cast_forward_inputs"):
            cast_fn = functools.partial(_cast_fp_tensor, self._mp_policy.param_dtype)
            args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
    if self._fsdp_param_group:
        _maybe_set_mccl_collectives(self)
        args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
    for fsdp_state in self._states_to_forward_prefetch:
        if (target_param_group := fsdp_state._fsdp_param_group) is not None:
            _maybe_set_mccl_collectives(fsdp_state)
            FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
    return args, kwargs


def _apply_custom_overlap_patch():
    FSDPState._init_shared_state = _fsdp_state_init_shared_state
    FSDPState._pre_forward = _pre_forward
    FSDPCommContext.lazy_init = comm_context_lazy_init
