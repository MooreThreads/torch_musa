"""Bucket layout planning for FSDP2 bucket-backed uneven parameter sharding.

A whole ``FSDPParamGroup`` is mapped into one global flat bucket. The bucket
(not each parameter) is evenly sharded across the data-parallel shard ranks, so
one parameter's local shard is the intersection of its global interval with the
rank's bucket interval. That intersection may be uneven or empty while still
being expressible as ``Shard(0)``.

``BucketLayoutPlanner.plan()`` is the distributed entry point. Exactly one
source rank per complete DP mesh computes a rank-independent ``_GeometryPlan``;
the plan is then broadcast across replica leaders (for HSDP) and finally within
each shard group. Every rank materializes its own ``BucketLayout`` locally from
that shared geometry.
"""

from __future__ import annotations

# pylint: disable=C0103,R0902,R0913,R0914,R0917

import hashlib
import itertools
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.distributed as dist
from torch._prims_common import make_contiguous_strides_for

_BOUNDARY_AWARE_PLAN_CACHE_MAX_SIZE = 128
_BOUNDARY_AWARE_PLAN_CACHE: "OrderedDict[tuple, _GeometryPlan]" = OrderedDict()
_BOUNDARY_AWARE_PLAN_CACHE_LOCK = threading.Lock()

# Bound the exact fallback's four 2**N state arrays. This does not cap the
# deterministic heuristic search (including mandatory-parameter permutations).
_BOUNDARY_AWARE_EXACT_MAX_PARAMS = 16


__all__ = [
    "ParamSpec",
    "BucketParamLayout",
    "BucketLayout",
    "BucketLayoutPlanner",
    "align_up",
    "row_unit_of",
    "validate_layout_locally",
    "set_planner_selector",
    "get_layout_planner",
]


class LayoutPlannerProtocol(Protocol):
    """Common distributed planner interface consumed by FSDP bucket creation."""

    def plan(
        self,
        param_specs: Sequence["ParamSpec"],
        shard_group: dist.ProcessGroup,
        *,
        replica_group: Optional[dist.ProcessGroup] = None,
        module_fqn: str = "",
        device: Optional[torch.device] = None,
    ) -> "BucketLayout": ...


PlannerSelector = Callable[[Sequence["ParamSpec"]], LayoutPlannerProtocol]


def _default_planner_selector(
    param_specs: Sequence["ParamSpec"],
) -> LayoutPlannerProtocol:
    del param_specs
    return BucketLayoutPlanner()


_planner_selector: PlannerSelector = _default_planner_selector


def set_planner_selector(selector: Optional[PlannerSelector]) -> None:
    """Set the process-wide bucket-layout planner selector.

    The selector is still evaluated by the caller on every rank, but expensive
    geometry solving is performed only by the DP source rank inside ``plan()``.
    Passing ``None`` restores the default ``BucketLayoutPlanner``.
    """

    global _planner_selector

    if selector is None:
        _planner_selector = _default_planner_selector
        return
    if not callable(selector):
        raise TypeError(
            "planner selector must be callable or None, " f"got {type(selector)!r}"
        )
    _planner_selector = selector


def get_layout_planner(
    param_specs: Sequence["ParamSpec"],
) -> LayoutPlannerProtocol:
    """Return the layout planner selected for one FSDP parameter group."""

    planner = _planner_selector(param_specs)
    if planner is None:
        raise RuntimeError("layout planner selector returned None")
    if not hasattr(planner, "plan"):
        raise TypeError(
            "layout planner selector must return an object implementing plan(...)"
        )
    return planner


def align_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the next multiple of ``multiple``."""

    if multiple <= 0:
        raise ValueError(f"alignment must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def row_unit_of(size: Sequence[int]) -> int:
    """Number of elements in one dim-0 row, the atomic Shard(0) slice."""

    if len(size) <= 1:
        return 1
    return math.prod(size[1:])


@dataclass(frozen=True)
class ParamSpec:
    """Parameter geometry needed for bucket placement."""

    fqn: str
    size: torch.Size
    stride: Tuple[int, ...]


@dataclass
class BucketParamLayout:
    """Placement of one parameter inside the global bucket and this rank's shard."""

    param_index: int
    fqn: str

    orig_size: torch.Size
    orig_stride: Tuple[int, ...]
    numel: int

    row_unit: int

    # [global_offset, global_offset + numel) inside the global bucket.
    global_offset: int

    # Intersection with this rank's bucket interval, in global bucket coords.
    local_global_begin: int
    local_global_end: int

    # The same intersection, relative to the parameter.
    param_flat_begin: int
    param_flat_end: int

    # The same intersection, relative to this rank's local bucket.
    local_bucket_offset: int
    local_numel: int

    local_size: torch.Size
    local_stride: Tuple[int, ...]

    # Dim-0 row index of the local shard within the global parameter, for DCP.
    global_dim0_offset: int

    @property
    def is_empty(self) -> bool:
        """Whether this rank owns no element of the parameter."""

        return self.local_numel == 0

    @property
    def global_end(self) -> int:
        """Exclusive end of the parameter inside the global bucket."""

        return self.global_offset + self.numel


@dataclass
class BucketLayout:
    """Element offsets and shard geometry for one parameter-group bucket."""

    module_fqn: str

    # Rank is the rank inside ``shard_group``, not the global process rank.
    rank: int
    world_size: int

    bucket_numel: int
    local_numel: int

    # Rank-boundary alignment factor.
    chunk_size_factor: int

    param_layouts: List[BucketParamLayout]
    padding_ranges: List[Tuple[int, int]]

    @property
    def local_begin(self) -> int:
        """Inclusive start of this rank's bucket interval, in global coords."""

        return self.rank * self.local_numel

    @property
    def local_end(self) -> int:
        """Exclusive end of this rank's bucket interval, in global coords."""

        return (self.rank + 1) * self.local_numel

    @property
    def payload_numel(self) -> int:
        """Global bucket elements that carry real parameter data."""

        return sum(p.numel for p in self.param_layouts)

    @property
    def padding_numel(self) -> int:
        """Global bucket elements that are alignment padding."""

        return self.bucket_numel - self.payload_numel

    @property
    def padding_ratio(self) -> float:
        """Fraction of the global bucket wasted on alignment padding."""

        if self.bucket_numel == 0:
            return 0.0
        return self.padding_numel / self.bucket_numel


@dataclass(frozen=True)
class _GeometryPlan:
    """Rank-independent placement from which every local layout is derived."""

    local_numel: int
    order: Tuple[int, ...]
    offsets: Tuple[int, ...]
    chunk_size_factor: int


def _compute_param_specs_hash(
    specs: Sequence[ParamSpec],
    world_size: int,
    module_fqn: str,
) -> str:
    """Hash rank-invariant inputs required to interpret a geometry plan."""

    hasher = hashlib.sha256()
    hasher.update(f"{world_size}|{module_fqn}|{len(specs)}".encode())
    for index, spec in enumerate(specs):
        hasher.update(
            (f"|{index}:{spec.fqn}:{tuple(spec.size)}:{tuple(spec.stride)}").encode()
        )
    return hasher.hexdigest()


def _materialize_layout(
    specs: Sequence[ParamSpec],
    geometry: _GeometryPlan,
    rank: int,
    world_size: int,
    *,
    module_fqn: str,
) -> BucketLayout:
    """Build this shard rank's local layout from shared global geometry."""

    if len(geometry.order) != len(specs) or len(geometry.offsets) != len(specs):
        raise ValueError(
            "geometry parameter count does not match ParamSpecs: "
            f"order={len(geometry.order)}, offsets={len(geometry.offsets)}, "
            f"specs={len(specs)}"
        )

    bucket_numel = geometry.local_numel * world_size
    rank_begin = rank * geometry.local_numel
    rank_end = rank_begin + geometry.local_numel

    param_layouts: List[BucketParamLayout] = []
    padding_ranges: List[Tuple[int, int]] = []
    previous_end = 0

    for index, offset in zip(geometry.order, geometry.offsets):
        if index < 0 or index >= len(specs):
            raise ValueError(f"geometry contains invalid parameter index {index}")

        spec = specs[index]
        numel = spec.size.numel()
        row_unit = row_unit_of(spec.size)

        if offset < previous_end:
            raise ValueError(
                f"{spec.fqn}: geometry overlaps the previous parameter: "
                f"offset={offset}, previous_end={previous_end}"
            )
        if offset > previous_end:
            padding_ranges.append((previous_end, offset))

        param_end = offset + numel
        previous_end = param_end
        if param_end > bucket_numel:
            raise ValueError(
                f"{spec.fqn}: geometry runs past bucket: "
                f"end={param_end}, bucket_numel={bucket_numel}"
            )

        begin = max(offset, rank_begin)
        end = min(param_end, rank_end)
        local_numel = max(0, end - begin)

        if local_numel == 0:
            begin = end = offset
            if not spec.size:
                raise ValueError(
                    f"{spec.fqn}: scalar parameters cannot have an empty Shard(0)"
                )

        if local_numel % row_unit != 0:
            raise AssertionError(
                f"{spec.fqn}: local shard {local_numel} is not a multiple of "
                f"row_unit {row_unit}; bucket geometry is broken"
            )

        local_size = (
            torch.Size((local_numel // row_unit, *spec.size[1:]))
            if spec.size
            else spec.size
        )

        param_layouts.append(
            BucketParamLayout(
                param_index=index,
                fqn=spec.fqn,
                orig_size=spec.size,
                orig_stride=spec.stride,
                numel=numel,
                row_unit=row_unit,
                global_offset=offset,
                local_global_begin=begin,
                local_global_end=end,
                param_flat_begin=begin - offset,
                param_flat_end=end - offset,
                local_bucket_offset=begin - rank_begin if local_numel else 0,
                local_numel=local_numel,
                local_size=local_size,
                local_stride=make_contiguous_strides_for(local_size),
                global_dim0_offset=(begin - offset) // row_unit,
            )
        )

    if previous_end < bucket_numel:
        padding_ranges.append((previous_end, bucket_numel))

    layout = BucketLayout(
        module_fqn=module_fqn,
        rank=rank,
        world_size=world_size,
        bucket_numel=bucket_numel,
        local_numel=geometry.local_numel,
        chunk_size_factor=geometry.chunk_size_factor,
        param_layouts=param_layouts,
        padding_ranges=padding_ranges,
    )
    validate_layout_locally(layout)
    return layout


def _is_row_safe(
    start: int,
    numel: int,
    row_unit: int,
    local_numel: int,
    world_size: int,
) -> bool:
    """Whether every shard boundary crossing this parameter is Shard(0)-safe."""

    end = start + numel
    first_boundary_rank = start // local_numel + 1
    last_boundary_rank = (end - 1) // local_numel

    if first_boundary_rank > last_boundary_rank or first_boundary_rank >= world_size:
        return True

    first_boundary = first_boundary_rank * local_numel
    if (first_boundary - start) % row_unit != 0:
        return False

    # If the parameter crosses multiple boundaries, adjacent boundaries are
    # separated by local_numel, which must itself preserve the row residue.
    return first_boundary_rank == last_boundary_rank or local_numel % row_unit == 0


def _earliest_row_safe_start(
    cursor: int,
    numel: int,
    row_unit: int,
    local_numel: int,
    world_size: int,
) -> Optional[int]:
    """Return the earliest legal parameter start at or after ``cursor``."""

    bucket_numel = local_numel * world_size
    max_start = bucket_numel - numel
    if cursor > max_start:
        return None

    best_start = (
        cursor
        if _is_row_safe(cursor, numel, row_unit, local_numel, world_size)
        else None
    )

    for boundary_rank in range(1, world_size):
        boundary = boundary_rank * local_numel

        # Starting exactly at a rank boundary can be the earliest safe choice.
        if cursor <= boundary <= max_start and _is_row_safe(
            boundary, numel, row_unit, local_numel, world_size
        ):
            best_start = boundary if best_start is None else min(best_start, boundary)

        # If this is the first crossed boundary, the parameter start must have
        # the same row residue as that boundary. Only the earliest such start
        # inside this rank interval can be optimal.
        interval_begin = max(cursor, (boundary_rank - 1) * local_numel)
        if interval_begin >= boundary:
            continue

        candidate = interval_begin + ((boundary - interval_begin) % row_unit)
        if (
            candidate <= min(max_start, boundary - 1)
            and candidate + numel > boundary
            and _is_row_safe(candidate, numel, row_unit, local_numel, world_size)
        ):
            best_start = candidate if best_start is None else min(best_start, candidate)

    return best_start


def _required_local_alignment(
    numels: Sequence[int],
    row_units: Sequence[int],
    lower_local_numel: int,
) -> Tuple[int, int]:
    """Compute the minimal local capacity forced by multi-boundary parameters.

    Any parameter longer than two local buckets must cross at least two rank
    boundaries. Their spacing is ``local_numel``, so that spacing must be a
    multiple of the parameter's row unit. The mandatory set can change after
    alignment, hence this monotone fixed-point computation.
    """

    local_numel = lower_local_numel
    mandatory_alignment = 1

    # The first pass collects constraints at the lower capacity. Increasing
    # local_numel can only remove mandatory parameters; because the alignment
    # is cumulative, a later pass only confirms that the result is stable.
    while True:
        for numel, row_unit in zip(numels, row_units):
            if numel > 2 * local_numel:
                mandatory_alignment = math.lcm(mandatory_alignment, row_unit)

        aligned_local_numel = align_up(lower_local_numel, mandatory_alignment)
        if aligned_local_numel == local_numel:
            return local_numel, mandatory_alignment
        local_numel = aligned_local_numel


def _dedupe_orders(orders: Iterable[Tuple[int, ...]]) -> Tuple[Tuple[int, ...], ...]:
    """Deduplicate deterministic order seeds while preserving insertion order."""

    result: List[Tuple[int, ...]] = []
    seen = set()
    for order in orders:
        if order in seen:
            continue
        seen.add(order)
        result.append(order)
    return tuple(result)


def _iter_boundary_aware_candidate_orders(
    numels: Sequence[int],
    row_units: Sequence[int],
    local_numel: int,
) -> Iterable[Tuple[int, ...]]:
    """Yield deterministic heuristic parameter orders for boundary-aware placement.

    The candidates include three whole-parameter orders, plus every mandatory
    parameter permutation combined with four remaining-parameter orders, with
    the mandatory block placed either before or after the remaining block.
    """

    num_params = len(numels)
    original_order = tuple(range(num_params))
    mandatory_params = tuple(
        index for index, numel in enumerate(numels) if numel > 2 * local_numel
    )
    mandatory_set = set(mandatory_params)
    remaining_params = tuple(
        index for index in original_order if index not in mandatory_set
    )

    remaining_orders = _dedupe_orders(
        (
            remaining_params,
            tuple(sorted(remaining_params, key=lambda index: (-numels[index], index))),
            tuple(
                sorted(
                    remaining_params,
                    key=lambda index: (-row_units[index], -numels[index], index),
                )
            ),
            tuple(
                sorted(
                    remaining_params,
                    key=lambda index: (row_units[index], -numels[index], index),
                )
            ),
        )
    )

    base_orders = _dedupe_orders(
        (
            original_order,
            tuple(sorted(original_order, key=lambda index: (-numels[index], index))),
            tuple(
                sorted(
                    original_order,
                    key=lambda index: (-row_units[index], -numels[index], index),
                )
            ),
        )
    )
    yield from base_orders

    # Keep the existing search semantics. Note that
    # _BOUNDARY_AWARE_EXACT_MAX_PARAMS does not cap this permutation stage.
    for mandatory_order in itertools.permutations(mandatory_params):
        for remaining_order in remaining_orders:
            yield mandatory_order + remaining_order
            yield remaining_order + mandatory_order


def _try_place_fixed_order(
    order: Sequence[int],
    numels: Sequence[int],
    row_units: Sequence[int],
    local_numel: int,
    world_size: int,
) -> Optional[Tuple[int, Tuple[int, ...]]]:
    """Greedily place one fixed order at each parameter's earliest legal start."""

    cursor = 0
    offsets: List[int] = []
    for index in order:
        start = _earliest_row_safe_start(
            cursor,
            numels[index],
            row_units[index],
            local_numel,
            world_size,
        )
        if start is None:
            return None
        offsets.append(start)
        cursor = start + numels[index]
    return cursor, tuple(offsets)


def _exact_subset_placement(
    numels: Sequence[int],
    row_units: Sequence[int],
    local_numel: int,
    world_size: int,
) -> Optional[Tuple[int, Tuple[int, ...], Tuple[int, ...]]]:
    """Exact subset-DP fallback for small parameter groups."""

    num_params = len(numels)
    bucket_numel = local_numel * world_size
    num_states = 1 << num_params
    unreachable = bucket_numel + 1

    cursors = [unreachable] * num_states
    previous_masks = [-1] * num_states
    selected_params = [-1] * num_states
    selected_offsets = [-1] * num_states
    cursors[0] = 0

    for mask, cursor in enumerate(cursors):
        if cursor == unreachable:
            continue

        for index, (numel, row_unit) in enumerate(zip(numels, row_units)):
            if mask & (1 << index):
                continue

            start = _earliest_row_safe_start(
                cursor,
                numel,
                row_unit,
                local_numel,
                world_size,
            )
            if start is None:
                continue

            next_mask = mask | (1 << index)
            next_cursor = start + numel
            if next_cursor < cursors[next_mask]:
                cursors[next_mask] = next_cursor
                previous_masks[next_mask] = mask
                selected_params[next_mask] = index
                selected_offsets[next_mask] = start

    if cursors[-1] == unreachable:
        return None

    order: List[int] = []
    offsets: List[int] = []
    mask = num_states - 1
    while mask:
        order.append(selected_params[mask])
        offsets.append(selected_offsets[mask])
        mask = previous_masks[mask]

    order.reverse()
    offsets.reverse()
    return cursors[-1], tuple(order), tuple(offsets)


def _get_cached_geometry(cache_key: tuple) -> Optional[_GeometryPlan]:
    with _BOUNDARY_AWARE_PLAN_CACHE_LOCK:
        geometry = _BOUNDARY_AWARE_PLAN_CACHE.get(cache_key)
        if geometry is not None:
            _BOUNDARY_AWARE_PLAN_CACHE.move_to_end(cache_key)
        return geometry


def _put_cached_geometry(cache_key: tuple, geometry: _GeometryPlan) -> None:
    with _BOUNDARY_AWARE_PLAN_CACHE_LOCK:
        _BOUNDARY_AWARE_PLAN_CACHE[cache_key] = geometry
        _BOUNDARY_AWARE_PLAN_CACHE.move_to_end(cache_key)
        while len(_BOUNDARY_AWARE_PLAN_CACHE) > _BOUNDARY_AWARE_PLAN_CACHE_MAX_SIZE:
            _BOUNDARY_AWARE_PLAN_CACHE.popitem(last=False)


class BucketLayoutPlanner:
    """Plan a zero-copy FSDP bucket using Shard(0)-safe rank boundaries."""

    def plan(
        self,
        param_specs: Sequence[ParamSpec],
        shard_group: dist.ProcessGroup,
        *,
        replica_group: Optional[dist.ProcessGroup] = None,
        module_fqn: str = "",
        device: Optional[torch.device] = None,
    ) -> BucketLayout:
        """Plan once per complete DP mesh and return this rank's local layout.

        ``shard_group`` defines the bucket sharding dimension and therefore the
        layout ``rank`` / ``world_size``. If ``replica_group`` is provided, only
        coordinate ``(replica=0, shard=0)`` solves the geometry. The geometry is
        first broadcast among shard-group leaders through ``replica_group`` and
        then from each leader to its own ``shard_group``.

        For accelerator-only collective backends, pass ``device`` explicitly so
        object collectives and validation collectives use the correct device.
        """

        if len(param_specs) == 0:
            raise ValueError("cannot plan a bucket for an empty parameter group")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "BucketLayoutPlanner.plan() requires initialized torch.distributed; "
                "geometry-only tests may call _plan_geometry() directly"
            )
        if shard_group is None:
            raise ValueError("shard_group must be provided")

        shard_rank = dist.get_rank(group=shard_group)
        shard_world_size = dist.get_world_size(group=shard_group)
        if shard_rank < 0:
            raise RuntimeError("current process is not a member of shard_group")
        if shard_world_size <= 0:
            raise RuntimeError(f"invalid shard_group size {shard_world_size}")

        replica_rank = 0
        if replica_group is not None:
            replica_rank = dist.get_rank(group=replica_group)
            if replica_rank < 0:
                raise RuntimeError("current process is not a member of replica_group")

        is_source = shard_rank == 0 and (replica_group is None or replica_rank == 0)

        # (status, geometry, param_metadata_hash, error_type, error_message)
        payload = [(None, None, None, None, None)]
        source_error: Optional[BaseException] = None

        if is_source:
            try:
                geometry = self._plan_geometry(param_specs, shard_world_size)
                metadata_hash = _compute_param_specs_hash(
                    param_specs,
                    shard_world_size,
                    module_fqn,
                )
                payload[0] = ("ok", geometry, metadata_hash, None, None)
            except Exception as error:  # pylint: disable=broad-exception-caught
                source_error = error
                payload[0] = (
                    "error",
                    None,
                    None,
                    type(error).__name__,
                    str(error),
                )

        # For HSDP, only shard-group leaders participate in this replica-group
        # broadcast. It propagates one geometry from the unique DP source to
        # every shard-group leader.
        if replica_group is not None and shard_rank == 0:
            dist.broadcast_object_list(
                payload,
                group=replica_group,
                group_src=0,
                device=device,
            )

        # Each shard-group leader now fans the same geometry out to its shards.
        dist.broadcast_object_list(
            payload,
            group=shard_group,
            group_src=0,
            device=device,
        )

        status, geometry, source_metadata_hash, error_type, error_message = payload[0]
        if status != "ok":
            if source_error is not None:
                raise source_error
            raise RuntimeError(
                "zero-copy bucket layout planning failed on the DP source rank: "
                f"{error_type}: {error_message}"
            )

        layout: Optional[BucketLayout] = None
        local_error: Optional[BaseException] = None
        try:
            local_metadata_hash = _compute_param_specs_hash(
                param_specs,
                shard_world_size,
                module_fqn,
            )
            if local_metadata_hash != source_metadata_hash:
                raise ValueError(
                    "zero-copy parameter metadata differs across the DP mesh"
                )

            layout = _materialize_layout(
                param_specs,
                geometry,
                shard_rank,
                shard_world_size,
                module_fqn=module_fqn,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            local_error = error

        # Make rank-local metadata/materialization failures mesh-wide. This
        # prevents successful ranks from leaving planning while another rank
        # fails and later deadlocks in FSDP collectives.
        success = torch.tensor(
            int(local_error is None),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(success, op=dist.ReduceOp.MIN, group=shard_group)
        if replica_group is not None:
            dist.all_reduce(success, op=dist.ReduceOp.MIN, group=replica_group)

        if not success.item():
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                "zero-copy bucket layout validation failed on another DP rank"
            )

        assert layout is not None
        return layout

    def _plan_geometry(
        self,
        param_specs: Sequence[ParamSpec],
        world_size: int,
    ) -> _GeometryPlan:
        """Pure local geometry solver used only by the DP source rank."""

        return self._plan_boundary_aware_geometry(param_specs, world_size)

    @staticmethod
    def _plan_fixed_order_geometry(
        param_specs: Sequence[ParamSpec],
        world_size: int,
    ) -> _GeometryPlan:
        """Conservative fixed-order fallback using a global row-unit LCM."""

        row_units = [row_unit_of(spec.size) for spec in param_specs]
        chunk_size_factor = 1
        for unit in row_units:
            chunk_size_factor = math.lcm(chunk_size_factor, unit)

        cursor = 0
        global_offsets: List[int] = []
        for spec, row_unit in zip(param_specs, row_units):
            aligned = align_up(cursor, row_unit)
            global_offsets.append(aligned)
            cursor = aligned + spec.size.numel()

        bucket_numel = align_up(cursor, world_size * chunk_size_factor)
        local_numel = bucket_numel // world_size

        return _GeometryPlan(
            local_numel=local_numel,
            order=tuple(range(len(param_specs))),
            offsets=tuple(global_offsets),
            chunk_size_factor=chunk_size_factor,
        )

    @staticmethod
    def _plan_boundary_aware_geometry(
        param_specs: Sequence[ParamSpec],
        world_size: int,
    ) -> _GeometryPlan:
        """Plan using only row constraints of parameters that cross boundaries."""

        row_units = tuple(row_unit_of(spec.size) for spec in param_specs)
        numels = tuple(spec.size.numel() for spec in param_specs)
        payload_numel = sum(numels)
        lower_local_numel = (payload_numel + world_size - 1) // world_size

        # Align local capacity for parameters forced to cross multiple rank boundaries;
        # later placement ensures those boundaries fall between parameter rows.
        local_numel, mandatory_alignment = _required_local_alignment(
            numels,
            row_units,
            lower_local_numel,
        )
        bucket_numel = local_numel * world_size

        cache_key = (
            world_size,
            tuple(tuple(spec.size) for spec in param_specs),
        )
        cached_geometry = _get_cached_geometry(cache_key)
        if cached_geometry is not None:
            return cached_geometry

        best_placement: Optional[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = None

        for order in _iter_boundary_aware_candidate_orders(
            numels,
            row_units,
            local_numel,
        ):
            placed = _try_place_fixed_order(
                order,
                numels,
                row_units,
                local_numel,
                world_size,
            )
            if placed is None:
                continue

            cursor, offsets = placed
            placement = (cursor, tuple(order), offsets)
            if best_placement is None or placement[:2] < best_placement[:2]:
                best_placement = placement

        if (
            best_placement is None
            and len(param_specs) <= _BOUNDARY_AWARE_EXACT_MAX_PARAMS
        ):
            best_placement = _exact_subset_placement(
                numels,
                row_units,
                local_numel,
                world_size,
            )

        if best_placement is None:
            geometry = BucketLayoutPlanner._plan_fixed_order_geometry(
                param_specs,
                world_size,
            )
            _put_cached_geometry(cache_key, geometry)
            return geometry

        cursor, order, offsets = best_placement
        if cursor > bucket_numel:
            raise AssertionError(
                "boundary-aware placement exceeds the bucket despite successful "
                f"placement: cursor={cursor}, bucket_numel={bucket_numel}"
            )

        geometry = _GeometryPlan(
            local_numel=local_numel,
            order=order,
            offsets=offsets,
            chunk_size_factor=mandatory_alignment,
        )
        _put_cached_geometry(cache_key, geometry)
        return geometry


def validate_layout_locally(layout: BucketLayout) -> None:
    """Assert internal consistency invariants of a materialized bucket layout."""

    assert (
        layout.bucket_numel % layout.world_size == 0
    ), "bucket is not evenly shardable"
    assert layout.local_numel * layout.world_size == layout.bucket_numel
    assert layout.chunk_size_factor > 0
    assert layout.local_numel % layout.chunk_size_factor == 0

    covered = 0
    prev_end = 0
    for p in layout.param_layouts:
        assert p.global_offset >= prev_end, f"{p.fqn}: parameters overlap in the bucket"
        assert p.global_end <= layout.bucket_numel, f"{p.fqn}: runs past the bucket"
        prev_end = p.global_end
        covered += p.numel

        assert 0 <= p.local_numel <= layout.local_numel
        assert p.local_bucket_offset + p.local_numel <= layout.local_numel
        assert math.prod(p.local_size) == p.local_numel, f"{p.fqn}: local_size mismatch"
        if p.local_numel:
            assert p.local_global_begin == layout.local_begin + p.local_bucket_offset
            assert p.local_global_end - p.local_global_begin == p.local_numel
            assert p.param_flat_begin % p.row_unit == 0

        for boundary_rank in range(1, layout.world_size):
            boundary = boundary_rank * layout.local_numel
            if p.global_offset < boundary < p.global_end:
                assert (
                    boundary - p.global_offset
                ) % p.row_unit == 0, f"{p.fqn}: rank boundary {boundary} splits a row"

    assert covered == layout.payload_numel
    padding = sum(end - begin for begin, end in layout.padding_ranges)
    assert padding == layout.padding_numel, "padding ranges do not cover the padding"
    for begin, end in layout.padding_ranges:
        assert 0 <= begin < end <= layout.bucket_numel
