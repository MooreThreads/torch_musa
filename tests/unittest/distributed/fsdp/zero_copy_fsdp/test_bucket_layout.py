"""tests for zero-copy FSDP bucket layout planning."""

from contextlib import nullcontext
from unittest import mock
import time

import torch
import torch.distributed as dist
from torch._prims_common import make_contiguous_strides_for

from torch_musa.distributed._composable.fsdp.zero_copy import _bucket_layout
from torch_musa.distributed._composable.fsdp.zero_copy._bucket_layout import (
    BucketLayoutPlanner,
    ParamSpec,
)
from torch_musa.testing.common_dist import MultiProcessingTest

_WORLD_SIZE = 4
_SHARD_GROUPS = ((0, 1), (2, 3))
_REPLICA_GROUPS = ((0, 2), (1, 3))


def _small_specs():
    shapes = (
        ("self_attn.q_proj.weight", (4096, 4096)),
        ("self_attn.k_proj.weight", (1024, 4096)),
        ("self_attn.v_proj.weight", (1024, 4096)),
        ("self_attn.o_proj.weight", (4096, 4096)),
        ("mlp.gate_proj.weight", (14336, 4096)),
        ("mlp.up_proj.weight", (14336, 4096)),
        ("mlp.down_proj.weight", (4096, 14336)),
        ("input_layernorm.weight", (4096,)),
    )
    return [
        ParamSpec(name, torch.Size(shape), make_contiguous_strides_for(shape))
        for name, shape in shapes
    ]


def _layout_signature(layout):
    return (
        layout.bucket_numel,
        layout.local_numel,
        layout.chunk_size_factor,
        tuple(
            (p.param_index, p.fqn, p.numel, p.row_unit, p.global_offset)
            for p in layout.param_layouts
        ),
        tuple(layout.padding_ranges),
    )


class TestBucketLayout(MultiProcessingTest):
    """Distributed protocol tests for the bucket layout planner."""

    @property
    def world_size(self):
        return _WORLD_SIZE

    def _groups(self, hsdp):
        if not hsdp:
            return self.process_group, None

        shard_groups = [
            dist.new_group(list(ranks), backend=self.backend) for ranks in _SHARD_GROUPS
        ]
        replica_groups = [
            dist.new_group(list(ranks), backend=self.backend)
            for ranks in _REPLICA_GROUPS
        ]
        dist.barrier()
        return shard_groups[self.rank // 2], replica_groups[self.rank % 2]

    def _assert_layout(self, layout, shard_group):
        shard_rank = dist.get_rank(group=shard_group)
        shard_world_size = dist.get_world_size(group=shard_group)
        self.assertEqual(layout.rank, shard_rank)
        self.assertEqual(layout.world_size, shard_world_size)
        self.assertEqual(layout.bucket_numel, layout.local_numel * shard_world_size)

        for placement in layout.param_layouts:
            begin = max(placement.global_offset, layout.local_begin)
            end = min(placement.global_end, layout.local_end)
            self.assertEqual(placement.local_numel, max(0, end - begin))

    def _plan_success(self, hsdp):
        shard_group, replica_group = self._groups(hsdp)
        layout = BucketLayoutPlanner().plan(
            _small_specs(),
            shard_group,
            replica_group=replica_group,
            module_fqn="block",
            device=torch.device("musa"),
        )
        self._assert_layout(layout, shard_group)

        signatures = [None] * self.world_size
        dist.all_gather_object(signatures, _layout_signature(layout))
        self.assertTrue(all(signature == signatures[0] for signature in signatures))
        dist.barrier()

    def test_plan_success_dp_and_hsdp(self):
        """Planning works for plain shard DP and 2D shard/replica DP."""
        self._plan_success(hsdp=False)
        self._plan_success(hsdp=True)

    def _assert_failure_propagates(self, hsdp, failure_kind):
        shard_group, replica_group = self._groups(hsdp)
        specs = _small_specs()
        planner = BucketLayoutPlanner()

        if failure_kind == "metadata" and self.rank == _WORLD_SIZE - 1:
            shape = (specs[0].size[0] + 1, *specs[0].size[1:])
            specs[0] = ParamSpec(
                specs[0].fqn,
                torch.Size(shape),
                make_contiguous_strides_for(shape),
            )

        if failure_kind == "planning":
            context = (
                mock.patch.object(
                    planner,
                    "_plan_geometry",
                    side_effect=ValueError("injected planning failure"),
                )
                if self.rank == 0
                else nullcontext()
            )
        elif failure_kind == "local":
            context = (
                mock.patch.object(
                    _bucket_layout,
                    "validate_layout_locally",
                    side_effect=ValueError("injected local validation failure"),
                )
                if self.rank == _WORLD_SIZE - 1
                else nullcontext()
            )
        else:
            context = nullcontext()

        failed = False
        with context:
            try:
                planner.plan(
                    specs,
                    shard_group,
                    replica_group=replica_group,
                    module_fqn="block",
                    device=torch.device("musa"),
                )
            except (RuntimeError, ValueError):
                failed = True

        all_failed = torch.tensor(int(failed), dtype=torch.int32, device="musa")
        dist.all_reduce(all_failed, op=dist.ReduceOp.MIN)
        self.assertEqual(all_failed.item(), 1)
        dist.barrier()

    def test_plan_failures(self):
        """Source and rank-local failures reach every DP topology rank."""
        for hsdp in (False, True):
            for failure_kind in ("planning", "metadata", "local"):
                self._assert_failure_propagates(hsdp, failure_kind)

    def test_sparse_expert_padding(self):
        """Boundary-aware placement avoids the fixed-order global LCM padding."""
        if dist.get_rank() != 0:
            time.sleep(2)
            return
        shapes = [
            (768, 2048),
            (768,),
            (5120, 768),
            (576, 2048),
            (512,),
            (8960, 512),
            (2048, 5120),
            (64, 3072, 2048),
            (64, 2048, 1536),
            (64, 2048),
            (1536, 2048),
            (1536, 2048),
            (2048, 1536),
            (2048,),
            (2048,),
        ]
        specs = [
            ParamSpec(
                f"param_{index}",
                torch.Size(shape),
                make_contiguous_strides_for(shape),
            )
            for index, shape in enumerate(shapes)
        ]
        planner = BucketLayoutPlanner()

        fixed_geometry = planner._plan_fixed_order_geometry(specs, world_size=8)
        optimized_geometry = planner._plan_geometry(specs, world_size=8)
        fixed_layout = _bucket_layout._materialize_layout(
            specs, fixed_geometry, rank=0, world_size=8, module_fqn="sparse_expert"
        )
        optimized_layout = _bucket_layout._materialize_layout(
            specs, optimized_geometry, rank=0, world_size=8, module_fqn="sparse_expert"
        )

        assert fixed_layout.padding_numel == 119_663_360
        assert optimized_layout.padding_numel == 19_000_064
        assert optimized_layout.padding_ratio < fixed_layout.padding_ratio
