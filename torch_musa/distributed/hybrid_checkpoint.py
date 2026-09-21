"""Experimental hybrid checkpointing support.

.. warning::
    This feature and its public APIs are experimental. API signatures,
    configuration defaults, and runtime behavior may change in future releases.

When CPU activation offload is enabled, wrapped layers must satisfy layer-level
FILO: the first layer that runs forward must be the last layer that runs
backward. This constraint concerns layer order, not operator order. Autograd
execution order is dynamic and cannot be validated at initialization, so
callers are responsible for preserving this invariant.
"""

import enum
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache, partial
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import torch
import torch.fx.traceback as fx_traceback
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    ActivationWrapper,
)
from torch.distributed.utils import _pack_kwargs, _unpack_kwargs
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from torch.utils.checkpoint import (
    SAC_IGNORED_OPS,
    SelectiveCheckpointContext,
    _is_compiling,
    _maybe_detach,
    _VersionWrapper,
)
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint


# Experimental public API. Keep this surface intentionally small.
__all__ = [
    "HybridCheckpointConfig",
    "apply_hybrid_checkpoint",
    "apply_hybrid_checkpoint_per_layer",
    "get_default_hybrid_checkpoint_offload_ops",
    "get_default_hybrid_checkpoint_ops",
]


class _HybridCheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.

    The following policies are supported:

    - ``{MUST,PREFER}_SAVE``: The operation's output will be saved during the forward
      pass and will not be recomputed during the backward pass
    - ``{MUST,PREFER}_RECOMPUTE``: The operation's output will not be saved during the
      forward pass and will be recomputed during the backward pass

    Use ``MUST_*`` over ``PREFER_*`` to indicate that the policy should not be overridden
    by other subsystems like `torch.compile`.

    .. note::
        A policy function that always returns ``PREFER_RECOMPUTE`` is
        equivalent to vanilla checkpointing.

        A policy function that returns ``PREFER_SAVE`` every op is
        NOT equivalent to not using checkpointing. Using such a policy would
        save additional tensors not limited to ones that are actually needed for
        gradient computation.
    """

    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3
    MUST_SAVE_OFFLOAD = 4


_HYBRID_OFFLOAD_HANDLER = None

_HYBRID_DISPATCH_DEBUG_ENABLED = (
    os.environ.get("HYBRID_CHECKPOINT_DEBUG_DISPATCH", "0") == "1"
)
_HYBRID_DISPATCH_DEBUG_DUMPED = False
_HYBRID_DISPATCH_COUNTS = defaultdict(int)


def _record_hybrid_dispatch(phase, func, args, kwargs, is_compiling):
    if not _HYBRID_DISPATCH_DEBUG_ENABLED or _HYBRID_DISPATCH_DEBUG_DUMPED:
        return

    tensor_label = "no_tensor"
    leaves, _ = tree_flatten((args, kwargs))
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            tensor_label = f"{type(leaf).__name__}@{leaf.device.type}"
            break

    stage = "compiling" if is_compiling else "runtime"
    _HYBRID_DISPATCH_COUNTS[(phase, stage, str(func), tensor_label)] += 1


def _dump_hybrid_dispatch_diagnostics():
    global _HYBRID_DISPATCH_DEBUG_DUMPED

    if not _HYBRID_DISPATCH_DEBUG_ENABLED or _HYBRID_DISPATCH_DEBUG_DUMPED:
        return
    _HYBRID_DISPATCH_DEBUG_DUMPED = True

    is_main_process = (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )
    if is_main_process:
        print("[HybridDispatch] first completed backward diagnostics:")
        if not _HYBRID_DISPATCH_COUNTS:
            print("[HybridDispatch] no dispatch calls recorded")
        for (phase, stage, func, tensor_label), count in sorted(
            _HYBRID_DISPATCH_COUNTS.items()
        ):
            print(
                f"[HybridDispatch][{phase}][{stage}] "
                f"{func} {tensor_label} count={count}"
            )

    _HYBRID_DISPATCH_COUNTS.clear()


@dataclass(frozen=True)
class _HybridOffloadEntry:
    func: Any
    storage_index: int


@dataclass
class _HybridOffloadBuffer:
    buffer: torch.Tensor
    bucket_bytes: int
    dtype: torch.dtype
    buffer_id: int


@dataclass
class _HybridOffloadTensorState:
    device: torch.device
    shape: torch.Size
    numel: int
    buffer_id: Optional[int] = None
    cpu_backup: Optional[torch.Tensor] = None

    def is_pinned(self) -> bool:
        if self.cpu_backup is not None:
            return self.cpu_backup.is_pinned()
        return self.buffer_id is not None


class _HybridOffloadHandler:
    def __init__(
        self,
        offload_bucket_bytes: Optional[Union[int, Sequence[int]]] = None,
        bucket_granularity_bytes: int = 2 * 1024 * 1024,
        offload_bucket_log_interval: Optional[int] = None,
    ):
        self.current_layer = 0

        self.storages: List[Dict[Any, List[Any]]] = []
        self.offload_infos: List[List[_HybridOffloadEntry]] = []
        self.tensor_refs: List[List[torch.Tensor]] = []

        self.offload_bucket_bytes: Optional[Tuple[int, ...]] = None
        self.bucket_granularity_bytes: int = 2 * 1024 * 1024
        self.offload_bucket_log_interval: Optional[int] = None
        self.bucket_free_lists: Dict[
            Tuple[torch.dtype, int], List[_HybridOffloadBuffer]
        ] = defaultdict(list)
        self.bucket_buffers_by_id: Dict[int, _HybridOffloadBuffer] = {}
        self.bucket_inflight_buffer_ids: Set[int] = set()
        self.next_bucket_buffer_id = 0
        self.completed_steps = 0
        self.bucket_allocations = 0
        self.bucket_hits = 0
        self.bucket_fallbacks = 0
        self.bucket_fallback_reasons: Dict[str, int] = defaultdict(int)
        self.pinned_tensor_allocations = 0
        self.pinned_bytes_allocated = 0
        self._reset_step_bucket_stats()

        self.main_stream = torch.musa.current_stream()
        self.d2h_stream = torch.musa.Stream()
        self.h2d_stream = torch.musa.Stream()
        self.configure_bucket_pool(
            offload_bucket_bytes=offload_bucket_bytes,
            bucket_granularity_bytes=bucket_granularity_bytes,
            offload_bucket_log_interval=offload_bucket_log_interval,
        )

    def reset(self):
        # Step-local checkpoint metadata can be dropped after backward, but the
        # persistent pinned bucket pool must survive across steps to avoid
        # reallocating pinned memory for variable-length activations.
        self.current_layer = 0
        self.storages = []
        self.tensor_refs = []
        self.offload_infos = []

    @staticmethod
    def _normalize_bucket_bytes_config(
        offload_bucket_bytes: Optional[Union[int, Sequence[int]]]
    ) -> Optional[Tuple[int, ...]]:
        if offload_bucket_bytes is None:
            return None

        if isinstance(offload_bucket_bytes, int):
            bucket_values = [offload_bucket_bytes]
        else:
            bucket_values = [int(bucket_bytes) for bucket_bytes in offload_bucket_bytes]

        assert len(bucket_values) > 0, "offload_bucket_bytes must not be empty"
        for bucket_bytes in bucket_values:
            assert bucket_bytes > 0, "offload_bucket_bytes must contain positive values"

        return tuple(sorted(set(bucket_values)))

    def configure_bucket_pool(
        self,
        offload_bucket_bytes: Optional[Union[int, Sequence[int]]] = None,
        bucket_granularity_bytes: Optional[int] = None,
        offload_bucket_log_interval: Optional[int] = None,
    ):
        """Configure pinned CPU bucket allocation and logging."""
        normalized_bucket_bytes = self._normalize_bucket_bytes_config(
            offload_bucket_bytes
        )
        if bucket_granularity_bytes is not None:
            assert bucket_granularity_bytes > 0, "bucket_granularity_bytes must be > 0"
        if offload_bucket_log_interval is not None:
            assert (
                offload_bucket_log_interval > 0
            ), "offload_bucket_log_interval must be > 0"

        if self.bucket_buffers_by_id and (
            (
                normalized_bucket_bytes is not None
                and normalized_bucket_bytes != self.offload_bucket_bytes
            )
            or (
                bucket_granularity_bytes is not None
                and bucket_granularity_bytes != self.bucket_granularity_bytes
            )
        ):
            warnings.warn(
                "Changing hybrid offload bucket config after buffers were created "
                "only affects future bucket selection and new buffers.",
                stacklevel=2,
            )

        if normalized_bucket_bytes is not None:
            self.offload_bucket_bytes = normalized_bucket_bytes
        if bucket_granularity_bytes is not None:
            self.bucket_granularity_bytes = int(bucket_granularity_bytes)
        if offload_bucket_log_interval is not None:
            self.offload_bucket_log_interval = int(offload_bucket_log_interval)

    def clear_bucket_pool(self):
        """Release cached pinned CPU buffers and reset pool statistics."""
        self.bucket_free_lists = defaultdict(list)
        self.bucket_buffers_by_id = {}
        self.bucket_inflight_buffer_ids = set()
        self.next_bucket_buffer_id = 0
        self.completed_steps = 0
        self.bucket_allocations = 0
        self.bucket_hits = 0
        self.bucket_fallbacks = 0
        self.bucket_fallback_reasons = defaultdict(int)
        self.pinned_tensor_allocations = 0
        self.pinned_bytes_allocated = 0
        self._reset_step_bucket_stats()

    def _reset_step_bucket_stats(self):
        self.step_bucket_allocations = 0
        self.step_bucket_hits = 0
        self.step_bucket_fallbacks = 0
        self.step_bucket_fallback_reasons = defaultdict(int)

    @staticmethod
    def _format_mib(capacity_bytes: int) -> str:
        return f"{capacity_bytes / (1024 * 1024):.1f}MiB"

    @staticmethod
    def _format_gib(capacity_bytes: int) -> str:
        return f"{capacity_bytes / (1024 * 1024 * 1024):.4f}GiB"

    def _record_bucket_fallback(self, reason: str):
        self.bucket_fallbacks += 1
        self.step_bucket_fallbacks += 1
        self.bucket_fallback_reasons[reason] += 1
        self.step_bucket_fallback_reasons[reason] += 1

    def _summarize_bucket_pool(self, limit: int = 3) -> str:
        if not self.bucket_buffers_by_id:
            return "none"

        bucket_summary = defaultdict(lambda: {"free": 0, "inflight": 0})
        for (dtype, bucket_bytes), free_buffers in self.bucket_free_lists.items():
            bucket_summary[(dtype, bucket_bytes)]["free"] += len(free_buffers)
        for buffer_id in self.bucket_inflight_buffer_ids:
            bucket_buffer = self.bucket_buffers_by_id[buffer_id]
            bucket_summary[(bucket_buffer.dtype, bucket_buffer.bucket_bytes)][
                "inflight"
            ] += 1

        ordered_buckets = sorted(
            bucket_summary.items(),
            key=lambda item: (
                item[1]["free"] + item[1]["inflight"],
                item[0][1],
                str(item[0][0]),
            ),
            reverse=True,
        )
        parts = []
        for (dtype, bucket_bytes), counts in ordered_buckets[:limit]:
            dtype_name = str(dtype).replace("torch.", "")
            parts.append(
                f"{dtype_name}:{self._format_mib(bucket_bytes)}"
                f"(free={counts['free']},inflight={counts['inflight']})"
            )
        return ", ".join(parts)

    def _collect_live_pinned_stats(self) -> Tuple[int, int]:
        live_bucket_tensor_count = len(self.bucket_buffers_by_id)
        live_bucket_bytes = sum(
            bucket_buffer.bucket_bytes
            for bucket_buffer in self.bucket_buffers_by_id.values()
        )
        return live_bucket_tensor_count, live_bucket_bytes

    def _maybe_print_bucket_stats(self):
        if (
            self.offload_bucket_bytes is None
            or self.offload_bucket_log_interval is None
            or self.offload_bucket_log_interval <= 0
        ):
            return

        if self.completed_steps % self.offload_bucket_log_interval != 0:
            return

        step_total = self.step_bucket_allocations + self.step_bucket_hits
        hit_rate = self.step_bucket_hits / step_total if step_total > 0 else 0.0
        total_free_buffers = sum(
            len(buffers) for buffers in self.bucket_free_lists.values()
        )
        total_inflight_buffers = len(self.bucket_inflight_buffer_ids)
        live_bucket_pinned_tensor_count, live_bucket_pinned_bytes = (
            self._collect_live_pinned_stats()
        )
        fallback_reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(self.step_bucket_fallback_reasons.items())
        )
        if not fallback_reasons:
            fallback_reasons = "none"
        print(
            "[HybridOffloadHandler] "
            f"step={self.completed_steps} "
            f"bucket_hits={self.step_bucket_hits} "
            f"new_bucket_allocations={self.step_bucket_allocations} "
            f"bucket_fallbacks={self.step_bucket_fallbacks} "
            f"bucket_pool_size={len(self.bucket_buffers_by_id)} "
            f"free_buffers={total_free_buffers} "
            f"inflight_buffers={total_inflight_buffers} "
            f"bucket_hit_rate={hit_rate:.4f} "
            f"live_bucket_pinned_tensors={live_bucket_pinned_tensor_count} "
            f"live_bucket_pinned_bytes={live_bucket_pinned_bytes} "
            f"live_bucket_pinned_mib={self._format_mib(live_bucket_pinned_bytes)} "
            f"live_bucket_pinned_gib={self._format_gib(live_bucket_pinned_bytes)} "
            f"allocated_bucket_pinned_tensors={self.pinned_tensor_allocations} "
            f"allocated_bucket_pinned_bytes={self.pinned_bytes_allocated} "
            "allocated_bucket_pinned_mib="
            f"{self._format_mib(self.pinned_bytes_allocated)} "
            "allocated_bucket_pinned_gib="
            f"{self._format_gib(self.pinned_bytes_allocated)} "
            f"fallback_reasons={fallback_reasons} "
            f"top_buckets={self._summarize_bucket_pool()}"
        )

    @staticmethod
    def _normalize_capacity_bytes(capacity_bytes: int, element_size: int) -> int:
        if capacity_bytes <= 0:
            return 0
        remainder = capacity_bytes % element_size
        if remainder == 0:
            return capacity_bytes
        return capacity_bytes + (element_size - remainder)

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 0:
            return value
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _make_cpu_backup(src_tensor, pin_memory=True):
        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        return cpu_backup

    @staticmethod
    def _should_use_bucket_buffer(src_tensor, pin_memory: bool) -> bool:
        return pin_memory and src_tensor.layout == torch.strided

    def _select_bucket_bytes(self, required_bytes: int, element_size: int) -> int:
        assert self.offload_bucket_bytes is not None
        for bucket_bytes in self.offload_bucket_bytes:
            normalized_bucket_bytes = self._normalize_capacity_bytes(
                bucket_bytes, element_size
            )
            if normalized_bucket_bytes >= required_bytes:
                return normalized_bucket_bytes

        dynamic_bucket_bytes = self._round_up_to_multiple(
            required_bytes, self.bucket_granularity_bytes
        )
        return self._normalize_capacity_bytes(dynamic_bucket_bytes, element_size)

    def _allocate_bucket_buffer(
        self, bucket_bytes: int, dtype: torch.dtype, element_size: int
    ) -> _HybridOffloadBuffer:
        capacity_numel = bucket_bytes // element_size
        buffer_id = self.next_bucket_buffer_id
        self.next_bucket_buffer_id += 1
        bucket_buffer = _HybridOffloadBuffer(
            buffer=torch.empty(
                capacity_numel,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            ),
            bucket_bytes=bucket_bytes,
            dtype=dtype,
            buffer_id=buffer_id,
        )
        self.bucket_buffers_by_id[buffer_id] = bucket_buffer
        self.pinned_tensor_allocations += 1
        self.pinned_bytes_allocated += bucket_bytes
        return bucket_buffer

    def _acquire_bucket_buffer(self, src_tensor) -> _HybridOffloadBuffer:
        required_bytes = src_tensor.numel() * src_tensor.element_size()
        bucket_bytes = self._select_bucket_bytes(
            required_bytes, src_tensor.element_size()
        )
        bucket_key = (src_tensor.dtype, bucket_bytes)
        free_buffers = self.bucket_free_lists[bucket_key]
        if free_buffers:
            bucket_buffer = free_buffers.pop()
            self.bucket_hits += 1
            self.step_bucket_hits += 1
        else:
            bucket_buffer = self._allocate_bucket_buffer(
                bucket_bytes,
                src_tensor.dtype,
                src_tensor.element_size(),
            )
            self.bucket_allocations += 1
            self.step_bucket_allocations += 1

        assert (
            bucket_buffer.buffer_id not in self.bucket_inflight_buffer_ids
        ), "Bucket buffer is already inflight"
        self.bucket_inflight_buffer_ids.add(bucket_buffer.buffer_id)
        return bucket_buffer

    @staticmethod
    def _buffer_view(
        bucket_buffer: _HybridOffloadBuffer, shape: torch.Size, numel: int
    ) -> torch.Tensor:
        return bucket_buffer.buffer.narrow(0, 0, numel).view(shape)

    def _release_step_bucket_buffers(self):
        # Buffers stay inflight for the whole step so asynchronous H2D reloads
        # cannot race with a later offload reusing the same pinned storage.
        for buffer_id in list(self.bucket_inflight_buffer_ids):
            bucket_buffer = self.bucket_buffers_by_id.get(buffer_id)
            if bucket_buffer is None:
                continue
            self.bucket_free_lists[
                (bucket_buffer.dtype, bucket_buffer.bucket_bytes)
            ].append(bucket_buffer)
        self.bucket_inflight_buffer_ids.clear()

    def offload(
        self,
        src_tensor,
        pin_memory: bool = True,
    ):
        """Copy one device tensor into pinned CPU backing storage."""
        if not isinstance(src_tensor, torch.Tensor):
            return src_tensor

        fallback_reason = None
        if self.offload_bucket_bytes is None:
            fallback_reason = "pool_disabled"
        elif not pin_memory:
            fallback_reason = "pin_memory_disabled"
        elif src_tensor.numel() == 0:
            fallback_reason = "empty_tensor"
        elif not self._should_use_bucket_buffer(src_tensor, pin_memory):
            fallback_reason = "non_strided"

        if fallback_reason is not None:
            self._record_bucket_fallback(fallback_reason)
            cpu_backup = self._make_cpu_backup(src_tensor, pin_memory=pin_memory)
            return _HybridOffloadTensorState(
                device=src_tensor.device,
                shape=src_tensor.size(),
                numel=src_tensor.numel(),
                cpu_backup=cpu_backup,
            )

        bucket_buffer = self._acquire_bucket_buffer(src_tensor)
        cpu_backup = self._buffer_view(
            bucket_buffer, src_tensor.size(), src_tensor.numel()
        )
        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        return _HybridOffloadTensorState(
            device=src_tensor.device,
            shape=src_tensor.size(),
            numel=src_tensor.numel(),
            buffer_id=bucket_buffer.buffer_id,
        )

    def _resolve_cpu_backup(self, state: _HybridOffloadTensorState) -> torch.Tensor:
        if state.cpu_backup is not None:
            return state.cpu_backup

        if state.buffer_id is None:
            self._record_bucket_fallback("missing_buffer_state")
            raise AssertionError("Offload state has no CPU backing storage")

        bucket_buffer = self.bucket_buffers_by_id.get(state.buffer_id)
        if bucket_buffer is None:
            self._record_bucket_fallback("missing_buffer_state")
            raise AssertionError("Pinned offload bucket buffer not found during reload")
        return self._buffer_view(bucket_buffer, state.shape, state.numel)

    def reload(self, state, non_blocking=None, reload_buffer=None):
        """Reload one offloaded tensor state onto its original device."""
        assert isinstance(state, _HybridOffloadTensorState)
        device = state.device
        cpu_backup = self._resolve_cpu_backup(state)

        if non_blocking is None:
            non_blocking = state.is_pinned()

        if reload_buffer is None:
            return cpu_backup.to(device, non_blocking=non_blocking)

        assert (
            cpu_backup.size() == reload_buffer.size()
        ), "Can't copy two buffers of different sizes!"

        reload_buffer.copy_(cpu_backup, non_blocking=non_blocking)

        return reload_buffer

    def version_wrapper_offload(self, version_wrapper: _VersionWrapper):
        if not isinstance(version_wrapper.val, torch.Tensor):
            return
        # CPU outputs do not need GPU offload; keeping them inline avoids
        # bumping the version counter via host-side copy_().
        if version_wrapper.val.device.type == "cpu":
            return
        version_wrapper.val = self.offload(version_wrapper.val)

    def version_wrapper_reload(self, version_wrapper: _VersionWrapper):
        if not isinstance(version_wrapper.val, _HybridOffloadTensorState):
            return

        version_wrapper.val = self.reload(version_wrapper.val)

    def bulk_offload_layer(self, layer_to_offload: int):
        """Offload all selected saved activations for one layer."""
        storage = self.storages[layer_to_offload]
        offload_info = self.offload_infos[layer_to_offload]

        with torch.musa.stream(self.d2h_stream):
            for entry in offload_info:
                saved_output = storage[entry.func][entry.storage_index]
                version_wrappers, _ = tree_flatten(saved_output)
                for version_wrapper in version_wrappers:
                    if not isinstance(version_wrapper, _VersionWrapper):
                        continue
                    if not isinstance(version_wrapper.val, torch.Tensor):
                        continue
                    self.version_wrapper_offload(version_wrapper)

    def bulk_reload_layer(self, layer_to_reload: int):
        storage = self.storages[layer_to_reload]
        offload_info = self.offload_infos[layer_to_reload]

        with torch.musa.stream(self.h2d_stream):
            for entry in offload_info:
                tree_map(
                    self.version_wrapper_reload,
                    storage[entry.func][entry.storage_index],
                )

    def prefetch_previous_layer(self, layer_index: int):
        """Prefetch the previous layer when recompute enters this layer."""
        if layer_index <= 0:
            return

        # This prefetch runs when the checkpoint recompute context is entered. At
        # that point, the current layer's FSDP pre-backward unshard/copy-out has
        # already been enqueued on the main stream. Start reloading the previous
        # layer only after that copy-out, then overlap the H2D copies with the
        # current layer's recompute and backward kernels.
        self.h2d_stream.wait_stream(self.main_stream)
        self.bulk_reload_layer(layer_index - 1)

    def layer_pre_forward_hook(self):
        if self.current_layer > 0:
            # ensure previous layer kernel are finished
            self.d2h_stream.wait_stream(self.main_stream)
            self.bulk_offload_layer(self.current_layer - 1)

    def layer_post_forward_hook(self):
        if self.current_layer > 0:
            self.main_stream.wait_stream(self.d2h_stream)
            # release previous layer reference tensor
            self.tensor_refs[self.current_layer - 1].clear()

        self.current_layer += 1

    def layer_pre_backward_hook(self):
        self.current_layer -= 1

        # The previous layer was prefetched while the next layer was being
        # recomputed. Make its activation visible on the main stream before the
        # current layer's FSDP pre-backward/recompute path consumes it.
        self.main_stream.wait_stream(self.h2d_stream)

    def layer_post_backward_hook(self):
        """Release completed layer state and recycle buffers after a step."""
        if 0 <= self.current_layer < len(self.tensor_refs):
            # Clear refs for the layer that just finished backward. This mainly
            # helps the last layer release lingering GPU refs earlier.
            # TODO(mingyuan.wang): Verify with a memory profiler that this reduces device memory.
            self.tensor_refs[self.current_layer].clear()

        if self.current_layer == 0:
            # The step is done. Wait for outstanding H2D work to finish before
            # recycling borrowed pinned buffers back into the per-bucket free
            # lists, otherwise the next step could reuse and overwrite a host
            # buffer that is still being read by an in-flight H2D copy.
            #
            # NOTE: `torch.musa.synchronize()` is intentionally conservative.
            # If profiling shows this is too heavy, `self.h2d_stream.synchronize()`
            # should be a lighter alternative because all H2D reloads are issued
            # on `self.h2d_stream`.
            self.main_stream.wait_stream(self.h2d_stream)
            torch.musa.synchronize()
            self._release_step_bucket_buffers()
            self.completed_steps += 1
            self._maybe_print_bucket_stats()
            _dump_hybrid_dispatch_diagnostics()
            self.reset()
            self._reset_step_bucket_stats()


def _get_hybrid_offload_handler() -> _HybridOffloadHandler:
    if _HYBRID_OFFLOAD_HANDLER is None:
        raise RuntimeError("Hybrid offload handler has not been initialized")
    return _HYBRID_OFFLOAD_HANDLER


def _set_hybrid_offload_handler(
    offload_bucket_bytes: Optional[Union[int, Sequence[int]]] = None,
    bucket_granularity_bytes: Optional[int] = None,
    offload_bucket_log_interval: Optional[int] = None,
):
    global _HYBRID_OFFLOAD_HANDLER

    if _HYBRID_OFFLOAD_HANDLER is not None:
        if _HYBRID_OFFLOAD_HANDLER.current_layer != 0:
            raise RuntimeError(
                "Cannot reconfigure hybrid checkpointing during an active step"
            )
        _HYBRID_OFFLOAD_HANDLER.reset()
        _HYBRID_OFFLOAD_HANDLER.configure_bucket_pool(
            offload_bucket_bytes=offload_bucket_bytes,
            bucket_granularity_bytes=bucket_granularity_bytes,
            offload_bucket_log_interval=offload_bucket_log_interval,
        )
        return

    _HYBRID_OFFLOAD_HANDLER = _HybridOffloadHandler(
        offload_bucket_bytes=offload_bucket_bytes,
        bucket_granularity_bytes=(
            2 * 1024 * 1024
            if bucket_granularity_bytes is None
            else bucket_granularity_bytes
        ),
        offload_bucket_log_interval=offload_bucket_log_interval,
    )


@lru_cache(maxsize=None)
def _get_out_argument_names(func) -> Tuple[str, ...]:
    if not func._schema.is_mutable:
        return ()

    return tuple(
        argument.name for argument in func._schema.arguments if argument.is_out
    )


class _HybridCachingTorchDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls):
        return True

    # Used together with _HybridCachedTorchDispatchMode to implement SAC.
    def __init__(
        self, policy_fn, storage, offload_info, tensor_ref, enable_offload=True
    ):  # pylint: disable=super-init-not-called
        self.policy_fn = policy_fn

        self.storage = storage
        self.offload_info = offload_info
        self.tensor_ref = tensor_ref

        self.offload_handler = None
        self.layer_index = None
        if enable_offload:
            self.offload_handler = _get_hybrid_offload_handler()
            self.layer_index = len(self.offload_handler.storages)
            self.offload_handler.storages.append(self.storage)
            self.offload_handler.offload_infos.append(self.offload_info)
            self.offload_handler.tensor_refs.append(self.tensor_ref)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        # is_recompute ensure we are in conventional forward
        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=False), func, *args, **kwargs
        )
        is_compiling = _is_compiling(func, args, kwargs)
        if _HYBRID_DISPATCH_DEBUG_ENABLED:
            _record_hybrid_dispatch("forward", func, args, kwargs, is_compiling)

        if is_compiling:
            # Overwrite each node's "recompute" tag to add in the user annotation.
            fx_traceback.current_meta["recompute"] = policy

        out = func(*args, **kwargs)

        any_ret_has_alias_info = any(
            ret.alias_info is not None for ret in func._schema.returns
        )
        # any_ret_has_alias_info = False

        if (
            policy
            in (
                _HybridCheckpointPolicy.MUST_SAVE,
                _HybridCheckpointPolicy.PREFER_SAVE,
                _HybridCheckpointPolicy.MUST_SAVE_OFFLOAD,
            )
            or is_compiling
        ):
            out_argument_names = _get_out_argument_names(func)
            # Inductor extern kernels use out variants and may reuse their
            # destination buffers later in the compiled graph. Keep independent
            # values for SAC so later in-place writes cannot corrupt the cached
            # activations used during backward.
            saved_out = (
                tree_map(
                    lambda x: (
                        _maybe_detach(x, any_ret_has_alias_info).clone()
                        if isinstance(x, torch.Tensor)
                        else x
                    ),
                    out,
                )
                if out_argument_names
                else out
            )
            if policy is _HybridCheckpointPolicy.MUST_SAVE_OFFLOAD:
                self.tensor_ref.append(saved_out)
                self.offload_info.append(
                    _HybridOffloadEntry(
                        func=func,
                        storage_index=len(self.storage[func]),
                    )
                )

            self.storage[func].append(
                tree_map(
                    lambda x: _VersionWrapper(_maybe_detach(x, any_ret_has_alias_info)),
                    saved_out,
                )
            )

        return out


class _HybridCachedTorchDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls):
        return True

    # Used together with _HybridCachedTorchDispatchMode to implement SAC.
    def __init__(
        self,
        policy_fn,
        storage,
        allow_cache_entry_mutation,
        offload_handler=None,
        layer_index=None,
    ):  # pylint: disable=super-init-not-called
        self.policy_fn = policy_fn
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation
        self.offload_handler = offload_handler
        self.layer_index = layer_index
        self._reload_started = False

    def __enter__(self):
        if (
            not self._reload_started
            and self.offload_handler is not None
            and self.layer_index is not None
        ):
            self.offload_handler.prefetch_previous_layer(self.layer_index)
            self._reload_started = True
        return super().__enter__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=True), func, *args, **kwargs
        )
        is_compiling = _is_compiling(func, args, kwargs)
        if _HYBRID_DISPATCH_DEBUG_ENABLED:
            _record_hybrid_dispatch("recompute", func, args, kwargs, is_compiling)

        if (
            policy
            in (
                _HybridCheckpointPolicy.MUST_SAVE,
                _HybridCheckpointPolicy.PREFER_SAVE,
                _HybridCheckpointPolicy.MUST_SAVE_OFFLOAD,
            )
            or is_compiling
        ):
            storage = self.storage.get(func)
            if storage is None:
                raise RuntimeError(
                    f"{func} encountered during backward, but not found in storage"
                )
            if len(storage) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            cached_out = tree_map(
                lambda x: x.get_val(self.allow_cache_entry_mutation), storage.pop(0)
            )
            out_argument_names = _get_out_argument_names(func)
            if out_argument_names:
                # Inductor may ignore an extern kernel's return value and read
                # its supplied out buffers directly. Populate those buffers
                # from the cache while still skipping the operation.
                out_arguments = tuple(kwargs[name] for name in out_argument_names)
                cached_out_arguments = (
                    cached_out if len(out_arguments) > 1 else (cached_out,)
                )
                tree_map(
                    lambda out_argument, cached_out_argument: out_argument.copy_(
                        cached_out_argument
                    ),
                    out_arguments,
                    cached_out_arguments,
                )
                out = out_arguments if len(out_arguments) > 1 else out_arguments[0]
            else:
                out = cached_out
        else:
            out = func(*args, **kwargs)
        return out


def _create_hybrid_checkpoint_contexts(
    policy_fn, allow_cache_entry_mutation=False, enable_offload=True
):
    storage = defaultdict(list)
    offload_info = []
    tensor_ref = []
    caching_mode = _HybridCachingTorchDispatchMode(
        policy_fn,
        storage,
        offload_info,
        tensor_ref,
        enable_offload=enable_offload,
    )
    return (
        caching_mode,
        _HybridCachedTorchDispatchMode(
            policy_fn,
            storage,
            allow_cache_entry_mutation,
            offload_handler=caching_mode.offload_handler,
            layer_index=caching_mode.layer_index,
        ),
    )


def _extract_tensor_leaves_from_pytree(tree):
    leaves, treespec = tree_flatten(tree)
    tensor_leaf_indices = []
    tensor_leaves = []
    for idx, leaf in enumerate(leaves):
        if isinstance(leaf, torch.Tensor):
            tensor_leaf_indices.append(idx)
            tensor_leaves.append(leaf)
    return leaves, treespec, tensor_leaf_indices, tensor_leaves


def _rebuild_pytree_with_tensor_leaves(
    leaves, treespec, tensor_leaf_indices, replacement_tensor_leaves
):
    rebuilt_leaves = list(leaves)
    for idx, tensor_leaf in zip(tensor_leaf_indices, replacement_tensor_leaves):
        rebuilt_leaves[idx] = tensor_leaf
    return tree_unflatten(rebuilt_leaves, treespec)


def _normalize_autograd_function_outputs(outputs):
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, tuple):
        return outputs
    return tuple(outputs)


def _restore_tensor_leaf_grad_semantics(reference_tensor_leaves, wrapped_tensor_leaves):
    restored_leaves = []
    for reference_leaf, wrapped_leaf in zip(
        reference_tensor_leaves, wrapped_tensor_leaves
    ):
        if reference_leaf.requires_grad:
            restored_leaves.append(wrapped_leaf)
        else:
            restored_leaves.append(wrapped_leaf.detach())
    return tuple(restored_leaves)


class _HybridOffloadHeadFunction(  # pylint: disable=abstract-method
    torch.autograd.Function
):
    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx, offload_handler: _HybridOffloadHandler, *tensor_leaves
    ):
        offload_handler.layer_pre_forward_hook()
        ctx.offload_handler = offload_handler
        if len(tensor_leaves) == 1:
            return tensor_leaves[0]
        return tensor_leaves

    @staticmethod
    def backward(ctx, *grad_outputs):
        offload_handler = ctx.offload_handler
        offload_handler.layer_post_backward_hook()
        return None, *grad_outputs


class _HybridOffloadTailFunction(  # pylint: disable=abstract-method
    torch.autograd.Function
):
    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx, offload_handler: _HybridOffloadHandler, *tensor_leaves
    ):
        offload_handler.layer_post_forward_hook()
        ctx.offload_handler = offload_handler
        if len(tensor_leaves) == 1:
            return tensor_leaves[0]
        return tensor_leaves

    @staticmethod
    def backward(ctx, *output_grads):
        offload_handler = ctx.offload_handler
        offload_handler.layer_pre_backward_hook()
        return None, *output_grads


class _HybridCheckpointWrapper(ActivationWrapper):
    def __init__(
        self,
        mod: nn.Module,
        enable_offload=True,
        **checkpoint_fn_kwargs,
    ):
        super().__init__(mod)
        self.enable_offload = enable_offload
        self.checkpoint_fn = partial(
            torch_utils_checkpoint,
            use_reentrant=False,
            **checkpoint_fn_kwargs,
        )

        self.hybrid_handler = (
            _get_hybrid_offload_handler() if self.enable_offload else None
        )

    @torch.compiler.disable(recursive=False)
    def forward(self, *args, **kwargs):
        # Hybrid activation checkpointing/offload is a training-only optimization.
        # In eval / no-grad mode there is no backward pass to drain the per-step
        # handler state, so bypass the wrapper entirely and run the wrapped
        # module directly.
        if (not self.training) or (not torch.is_grad_enabled()):
            return self._checkpoint_wrapped_module(*args, **kwargs)

        if self.enable_offload:
            input_tree = (args, kwargs)
            (
                input_leaves,
                input_treespec,
                input_tensor_leaf_indices,
                input_tensor_leaves,
            ) = _extract_tensor_leaves_from_pytree(input_tree)

            if len(input_tensor_leaves) > 0:
                wrapped_input_leaves = _normalize_autograd_function_outputs(
                    _HybridOffloadHeadFunction.apply(
                        self.hybrid_handler,
                        *input_tensor_leaves,
                    )
                )
                wrapped_input_leaves = _restore_tensor_leaf_grad_semantics(
                    input_tensor_leaves,
                    wrapped_input_leaves,
                )
                new_args, new_kwargs = _rebuild_pytree_with_tensor_leaves(
                    input_leaves,
                    input_treespec,
                    input_tensor_leaf_indices,
                    wrapped_input_leaves,
                )
            else:
                # No tensor leaf means HeadModule cannot participate in autograd.
                # Keep the original structure and fall back to direct hook semantics.
                self.hybrid_handler.layer_pre_forward_hook()
                new_args, new_kwargs = args, kwargs
        else:
            new_args, new_kwargs = args, kwargs

        if new_kwargs != {}:
            # Pack the args and kwargs
            flat_args, kwarg_keys = _pack_kwargs(*new_args, **new_kwargs)

            # Function that only takes (packed) args, but can unpack them
            # into the original args and kwargs for the checkpointed
            # function, and runs that function.
            def my_function(*inputs):
                # unpack back into args and kwargs
                unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
                # run original module
                return self._checkpoint_wrapped_module(
                    *unpacked_args, **unpacked_kwargs
                )

            # Pass the function that only takes packed args into reentrant
            # checkpoint API.
            output = self.checkpoint_fn(  # type: ignore[misc]
                my_function,
                *flat_args,
            )
        else:
            output = self.checkpoint_fn(  # type: ignore[misc]
                self._checkpoint_wrapped_module, *new_args, **new_kwargs
            )

        if not self.enable_offload:
            return output

        (
            output_leaves,
            output_treespec,
            output_tensor_leaf_indices,
            output_tensor_leaves,
        ) = _extract_tensor_leaves_from_pytree(output)

        if len(output_tensor_leaves) == 0:
            self.hybrid_handler.layer_post_forward_hook()
            return output

        wrapped_output_leaves = _normalize_autograd_function_outputs(
            _HybridOffloadTailFunction.apply(
                self.hybrid_handler,
                *output_tensor_leaves,
            )
        )
        wrapped_output_leaves = _restore_tensor_leaf_grad_semantics(
            output_tensor_leaves,
            wrapped_output_leaves,
        )
        return _rebuild_pytree_with_tensor_leaves(
            output_leaves,
            output_treespec,
            output_tensor_leaf_indices,
            wrapped_output_leaves,
        )


def _hybrid_checkpoint_wrapper(
    module: nn.Module,
    enable_offload=True,
    **checkpoint_fn_kwargs,
) -> nn.Module:
    return _HybridCheckpointWrapper(
        module,
        enable_offload=enable_offload,
        **checkpoint_fn_kwargs,
    )


try:
    _DEFAULT_MUSA_FLASH_SDPA = (
        torch.ops.aten._scaled_dot_product_attention_flash_musa.default
    )
except AttributeError:
    _DEFAULT_MUSA_FLASH_SDPA = None


# Selective activation checkpoint operator list. Matrix operators are useful
# model-specific extensions, but are not enabled without profiling.
_save_list = set()
if _DEFAULT_MUSA_FLASH_SDPA is not None:
    _save_list.add(_DEFAULT_MUSA_FLASH_SDPA)
# _save_list.add(torch.ops.aten.mm.default)
# _save_list.add(torch.ops.aten.addmm.default)  # Linear with bias
# _save_list.add(torch.ops.aten.addmm.out)


# Selective activation offload operator list. SDPA output is non-contiguous;
# offloading it directly avoids contention from an extra contiguous kernel.
_cpu_save_list = set()
if _DEFAULT_MUSA_FLASH_SDPA is not None:
    _cpu_save_list.add(_DEFAULT_MUSA_FLASH_SDPA)
# _cpu_save_list.add(torch.ops.aten.mm.default)
# _cpu_save_list.add(torch.ops.aten.addmm.default)  # Linear with bias
# _cpu_save_list.add(torch.ops.aten.addmm.out)


def get_default_hybrid_checkpoint_ops() -> Tuple[torch._ops.OpOverload, ...]:
    """Return the default ops whose activations are saved.

    .. warning::
        This API and the default operator set are experimental and may change.
    """

    return tuple(_save_list)


def get_default_hybrid_checkpoint_offload_ops() -> Tuple[torch._ops.OpOverload, ...]:
    """Return the default saved ops whose activations are offloaded to CPU.

    .. warning::
        This API and the default operator set are experimental and may change.
    """

    return tuple(_cpu_save_list)


@dataclass(frozen=True)
class HybridCheckpointConfig:
    """Configuration for selective activation saving and CPU offload.

    .. warning::
        This configuration API is experimental. Fields, defaults, and
        validation rules may change in future releases.

    ``save_list`` identifies operations whose outputs should be cached instead of
    recomputed. ``cpu_save_list`` must be a subset of ``save_list`` and identifies
    cached outputs that should be moved to pinned CPU memory between forward and
    backward. Leaving both fields as ``None`` selects MUSA Flash-SDPA for both.
    ``save_frequencies`` optionally maps a saved op to ``N`` so only every Nth
    occurrence is saved. Forward and recompute counts are tracked
    independently, matching selective-checkpoint replay order.

    ``num_layers`` limits wrapping to the first N children in the block container.
    ``None`` wraps every child.

    When ``cpu_save_list`` is non-empty, wrapped layers must satisfy layer-level
    FILO: the first layer that runs forward must be the last layer that runs
    backward. This dynamic layer order cannot be checked when the configuration
    is initialized; callers must preserve the invariant.
    """

    num_layers: Optional[int] = None
    save_list: Optional[Sequence[torch._ops.OpOverload]] = None
    cpu_save_list: Optional[Sequence[torch._ops.OpOverload]] = None
    save_frequencies: Optional[Mapping[torch._ops.OpOverload, int]] = None
    preserve_rng_state: bool = True
    allow_cache_entry_mutation: bool = False
    offload_bucket_bytes: Optional[Union[int, Sequence[int]]] = None
    offload_dynamic_bucket_granularity_bytes: int = (  # pylint: disable=invalid-name
        2 * 1024 * 1024
    )
    offload_bucket_log_interval: Optional[int] = None

    def __post_init__(self):
        if self.num_layers is not None and self.num_layers < 0:
            raise ValueError("num_layers must be non-negative or None")
        if self.offload_dynamic_bucket_granularity_bytes <= 0:
            raise ValueError(
                "offload_dynamic_bucket_granularity_bytes must be positive"
            )
        if (
            self.offload_bucket_log_interval is not None
            and self.offload_bucket_log_interval <= 0
        ):
            raise ValueError("offload_bucket_log_interval must be positive or None")

        save_ops, offload_ops = self.resolve_ops()
        for op in (*save_ops, *offload_ops):
            if not isinstance(op, torch._ops.OpOverload):
                raise TypeError(
                    "save_list and cpu_save_list must contain specific OpOverload "
                    f"objects, got {op!r}"
                )
        if not set(offload_ops).issubset(save_ops):
            raise ValueError("cpu_save_list must be a subset of save_list")

        save_frequencies = self.resolve_save_frequencies()
        for op, frequency in save_frequencies.items():
            if not isinstance(op, torch._ops.OpOverload):
                raise TypeError(
                    "save_frequencies keys must be specific OpOverload "
                    f"objects, got {op!r}"
                )
            if op not in save_ops:
                raise ValueError("save_frequencies keys must be a subset of save_list")
            if (
                isinstance(frequency, bool)
                or not isinstance(frequency, int)
                or frequency <= 0
            ):
                raise ValueError("save_frequencies values must be positive integers")

        if self.offload_bucket_bytes is not None:
            bucket_bytes = (
                (self.offload_bucket_bytes,)
                if isinstance(self.offload_bucket_bytes, int)
                else tuple(self.offload_bucket_bytes)
            )
            if not bucket_bytes or any(value <= 0 for value in bucket_bytes):
                raise ValueError("offload_bucket_bytes must contain positive values")

        object.__setattr__(self, "_resolved_save_ops", tuple(save_ops))
        object.__setattr__(self, "_resolved_offload_ops", tuple(offload_ops))
        object.__setattr__(
            self,
            "_resolved_save_frequencies",
            MappingProxyType(dict(save_frequencies)),
        )

    def resolve_ops(
        self,
    ) -> Tuple[
        Tuple[torch._ops.OpOverload, ...],
        Tuple[torch._ops.OpOverload, ...],
    ]:
        """Resolve and cache the activation save and CPU-offload operators."""

        if hasattr(self, "_resolved_save_ops"):
            return self._resolved_save_ops, self._resolved_offload_ops

        default_save_ops = get_default_hybrid_checkpoint_ops()
        default_offload_ops = get_default_hybrid_checkpoint_offload_ops()
        save_ops = tuple(default_save_ops if self.save_list is None else self.save_list)
        offload_ops = tuple(
            default_offload_ops if self.cpu_save_list is None else self.cpu_save_list
        )
        return save_ops, tuple(offload_ops)

    def resolve_save_frequencies(
        self,
    ) -> Mapping[torch._ops.OpOverload, int]:
        """Resolve and cache per-operator activation save frequencies."""

        if hasattr(self, "_resolved_save_frequencies"):
            return self._resolved_save_frequencies
        if self.save_frequencies is None:
            return {}
        if not isinstance(self.save_frequencies, Mapping):
            raise TypeError("save_frequencies must be a mapping or None")
        return dict(self.save_frequencies)


def _get_hybrid_policy(
    save_ops: Set[torch._ops.OpOverload],
    offload_ops: Set[torch._ops.OpOverload],
    save_frequencies: Mapping[torch._ops.OpOverload, int],
):
    forward_counts = defaultdict(int)
    recompute_counts = defaultdict(int)

    def policy(ctx, func, *_args, **_kwargs):
        if func in save_ops:
            frequency = save_frequencies.get(func)
            if frequency is not None:
                counts = recompute_counts if ctx.is_recompute else forward_counts
                counts[func] += 1
                if counts[func] % frequency != 0:
                    return _HybridCheckpointPolicy.PREFER_RECOMPUTE
            if func in offload_ops:
                return _HybridCheckpointPolicy.MUST_SAVE_OFFLOAD
            return _HybridCheckpointPolicy.MUST_SAVE
        return _HybridCheckpointPolicy.PREFER_RECOMPUTE

    return policy


def _wrap_hybrid_checkpoint_block(
    module: nn.Module,
    config: HybridCheckpointConfig,
    save_ops: Set[torch._ops.OpOverload],
    offload_ops: Set[torch._ops.OpOverload],
    save_frequencies: Mapping[torch._ops.OpOverload, int],
) -> nn.Module:
    enable_offload = bool(offload_ops)

    def context_fn():
        return _create_hybrid_checkpoint_contexts(
            _get_hybrid_policy(
                save_ops,
                offload_ops,
                save_frequencies,
            ),
            allow_cache_entry_mutation=config.allow_cache_entry_mutation,
            enable_offload=enable_offload,
        )

    return _hybrid_checkpoint_wrapper(
        module,
        context_fn=context_fn,
        preserve_rng_state=config.preserve_rng_state,
        enable_offload=enable_offload,
    )


def _prepare_hybrid_checkpoint(
    config: HybridCheckpointConfig,
) -> Tuple[
    Set[torch._ops.OpOverload],
    Set[torch._ops.OpOverload],
    Mapping[torch._ops.OpOverload, int],
]:
    if not isinstance(config, HybridCheckpointConfig):
        raise TypeError("config must be a HybridCheckpointConfig")

    save_ops_tuple, offload_ops_tuple = config.resolve_ops()
    save_ops = set(save_ops_tuple)
    offload_ops = set(offload_ops_tuple)
    save_frequencies = config.resolve_save_frequencies()
    if offload_ops:
        _set_hybrid_offload_handler(
            offload_bucket_bytes=config.offload_bucket_bytes,
            bucket_granularity_bytes=config.offload_dynamic_bucket_granularity_bytes,
            offload_bucket_log_interval=config.offload_bucket_log_interval,
        )
    return save_ops, offload_ops, save_frequencies


def apply_hybrid_checkpoint_per_layer(
    layer: nn.Module,
    config: HybridCheckpointConfig,
) -> nn.Module:
    """Wrap and return one layer with hybrid activation checkpointing.

    .. warning::
        This transformation API is experimental. Its wrapping behavior may
        change in future releases.

    The caller is responsible for assigning the returned wrapper back to the
    model. ``config.num_layers`` is not used by this per-layer interface.
    When CPU offload is enabled, all manually wrapped layers must satisfy
    layer-level FILO: the first layer that runs forward must be the last layer
    that runs backward. The order cannot be validated at initialization, so
    callers must preserve this requirement.

    Args:
        layer: A single model layer to wrap.
        config: Hybrid checkpoint policy and offload settings.

    Returns:
        The hybrid-checkpoint wrapper for ``layer``.
    """

    if isinstance(layer, _HybridCheckpointWrapper):
        raise ValueError("layer is already hybrid-checkpointed")
    save_ops, offload_ops, save_frequencies = _prepare_hybrid_checkpoint(config)
    return _wrap_hybrid_checkpoint_block(
        layer,
        config,
        save_ops,
        offload_ops,
        save_frequencies,
    )


def apply_hybrid_checkpoint(
    blocks: nn.Module,
    config: HybridCheckpointConfig,
) -> None:
    """Apply hybrid activation checkpointing to a block container.

    .. warning::
        This transformation API is experimental. Its wrapping behavior and
        supported model structures may change in future releases.

    The function mutates ``blocks`` by replacing its first
    ``config.num_layers`` children with checkpoint wrappers.
    When CPU offload is enabled, those layers must satisfy layer-level FILO: the
    first layer that runs forward must be the last layer that runs backward.
    The order cannot be validated at initialization, so callers must preserve
    this requirement.

    Args:
        blocks: Module container whose children are the repeated model blocks.
        config: Hybrid checkpoint policy and offload settings.
    """

    if not isinstance(config, HybridCheckpointConfig):
        raise TypeError("config must be a HybridCheckpointConfig")

    named_blocks = list(blocks.named_children())
    if not named_blocks:
        raise ValueError("blocks must contain at least one child module")

    num_layers = (
        len(named_blocks)
        if config.num_layers is None
        else min(config.num_layers, len(named_blocks))
    )
    if num_layers == 0:
        return

    save_ops, offload_ops, save_frequencies = _prepare_hybrid_checkpoint(config)

    for block_name, block in named_blocks[:num_layers]:
        if isinstance(block, _HybridCheckpointWrapper):
            raise ValueError(f"Block {block_name!r} is already hybrid-checkpointed")
        blocks.register_module(
            block_name,
            _wrap_hybrid_checkpoint_block(
                block,
                config,
                save_ops,
                offload_ops,
                save_frequencies,
            ),
        )
