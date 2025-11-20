# pylint: disable=W0611

"""Register Sharding rules or strategies for PyTorch DTensor operators"""

from typing import List, Optional, Sequence

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    register_op_strategy,
)
from torch.distributed.tensor._utils import normalize_to_torch_size
from torch.distributed.tensor._ops._math_ops import (
    _replicate_dims_start_at,
    map_placements_after_reduction,
    _infer_reduce_dims_map,
)


@register_op_strategy(
    [torch.ops.aten._fused_rmsnorm_forward.default],
    schema_info=RuntimeSchemaInfo(1),
)
def fused_rmsnorm_strategy(op_schema: OpSchema) -> OpStrategy:
    """DTensor sharding strategy of fused_rmsnorm_forward"""
    # args: input, normalized_shape, eps, weight (optional)
    assert len(op_schema.args_schema) == 4
    input_strategy, normalized_shape, _, weight_strategy = op_schema.args_schema

    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)

    # trailing on last # dims
    input_ndim = input_strategy.ndim
    axis = input_ndim - len(normalized_size)

    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []
        input_src_spec = input_placement_strategy.output_spec

        input_target_spec = DTensorSpec(
            mesh=input_strategy.mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_target_spec)
        )

        if weight_strategy:
            assert isinstance(weight_strategy, OpStrategy)
            weight_src_spec = weight_strategy.strategies[idx].output_spec
            weight_target_spec = DTensorSpec(
                mesh=weight_strategy.mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_target_spec)
            redistribute_costs.append(
                generate_redistribute_costs(weight_strategy, weight_target_spec)
            )

        output_target_spec = input_target_spec
        output_strategy.strategies.append(
            PlacementStrategy(
                output_specs=output_target_spec,
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_op_strategy(
    [torch.ops.aten._fused_rmsnorm_backward.default],
    schema_info=RuntimeSchemaInfo(3),
)
def fused_rmsnorm_bwd_strategy(op_schema: OpSchema) -> OpStrategy:
    """DTensor sharding strategy of fused_rmsnorm_backward"""
    # args: grad_out, invvar, input, normalized_shape, eps, weight
    assert len(op_schema.args_schema) == 6
    (
        grad_out_strategy,
        invvar_strategy,
        input_strategy,
        normalized_shape,
        _,
        weight_strategy,
    ) = op_schema.args_schema

    assert isinstance(grad_out_strategy, OpStrategy)
    assert isinstance(invvar_strategy, OpStrategy)
    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(weight_strategy, OpStrategy)

    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)
    input_ndim = input_strategy.ndim
    axis = input_ndim - len(normalized_size)
    outer_dims = list(range(axis))

    out_tuple_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        output_specs_list: List[Optional[DTensorSpec]] = []
        input_specs_list: List[DTensorSpec] = []
        redistribute_costs = []

        input_src_spec = input_placement_strategy.output_spec

        grad_out_target_spec = DTensorSpec(
            mesh=grad_out_strategy.mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        input_specs_list.append(grad_out_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(grad_out_strategy, grad_out_target_spec)
        )
        output_specs_list.append(grad_out_target_spec)

        # arg: invvar
        invvar_src_spec = invvar_strategy.strategies[idx].output_spec
        input_specs_list.append(invvar_src_spec)
        redistribute_costs.append([0.0 for _ in invvar_strategy.strategies])

        # arg: input
        input_target_spec = DTensorSpec(
            mesh=input_strategy.mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        input_specs_list.append(input_target_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_target_spec)
        )

        # arg: weight
        if weight_strategy is not None:
            # should we respect the input Spec of weight ?
            # weight_src_spec = weight_strategy.strategies[idx].output_spec
            # input_specs_list.append(weight_src_spec)
            # redistribute_costs.append([0.0 for _ in weight_strategy.strategies])

            weight_src_spec = weight_strategy.strategies[idx].output_spec
            weight_target_spec = DTensorSpec(
                mesh=weight_strategy.mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            input_specs_list.append(weight_target_spec)
            redistribute_costs.append(
                generate_redistribute_costs(weight_strategy, weight_target_spec)
            )

            inp_placements = _replicate_dims_start_at(input_src_spec.placements, axis)
            reduce_dims_map = _infer_reduce_dims_map(
                outer_dims, input_src_spec.ndim, False
            )
            out_placements = map_placements_after_reduction(
                inp_placements, outer_dims, reduce_dims_map, "sum"
            )
            weight_out_spec = DTensorSpec(
                mesh=weight_strategy.mesh,
                placements=out_placements,
                tensor_meta=weight_src_spec.tensor_meta,
            )
            output_specs_list.append(weight_out_spec)

        out_tuple_strategy.strategies.append(
            PlacementStrategy(
                output_specs=tuple(output_specs_list),
                input_specs=input_specs_list,
                redistribute_cost=redistribute_costs,
            )
        )

    return out_tuple_strategy
