# pylint: disable=W0611

"""Register Sharding rules or strategies for PyTorch DTensor operators"""


import torch
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    register_op_strategy,
)

from torch.distributed.tensor.placement_types import Replicate, Shard


# pylint: disable-all
@register_op_strategy(
    torch.ops.aten._scaled_dot_product_attention_flash_musa.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_attention_flash_musa_strategy(op_schema: OpSchema) -> OpStrategy:
    """DTensor sharding strategy of _scaled_dot_product_attention_flash_musa"""
    # args: query, key, value, attn_mask, dropout_p, is_causal, scale,

    mesh = op_schema.get_mesh_from_args()

    args = op_schema.args_schema
    n_args = len(args)

    query_strategy = args[0]  # query

    attn_mask_strategy = args[3] if n_args > 3 else None
    dropout_p = args[4] if n_args > 4 else 0.0

    assert isinstance(query_strategy, OpStrategy)

    query_shape = query_strategy.strategies[0].output_spec.shape
    is_4d = len(query_shape) == 4

    single_mesh_dim_strategies = []

    all_replicate: PlacementList = [
        Replicate(),  # output
        Replicate(),  # logsumexp
        Replicate(),  # dropout_mask
        Replicate(),  # rng_state
        Replicate(),  # query
        Replicate(),  # key
        Replicate(),  # value
    ]
    if attn_mask_strategy is not None:
        all_replicate.append(Replicate())  # attn_mask

    single_mesh_dim_strategies.append(all_replicate)

    if is_4d:
        # [batch_size, num_heads, seq_len, head_dim]
        tp_sharding = Shard(1)
    else:
        # [num_heads, seq_len, head_dim]
        tp_sharding = Shard(0)

    has_dropout = dropout_p > 0.0
    dropout_mask_sharding = tp_sharding if has_dropout else Replicate()

    num_heads_dim_sharding: PlacementList = [
        tp_sharding,  # output
        tp_sharding,  # logsumexp
        dropout_mask_sharding,  # dropout_mask
        Replicate(),  # rng_state
        tp_sharding,  # query
        tp_sharding,  # key
        tp_sharding,  # value
    ]
    if attn_mask_strategy is not None:
        num_heads_dim_sharding.append(Replicate())

    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Context Parallelism: shards on the sequence dim
    if is_4d:
        seq_dim_sharding = Shard(2)
    else:
        seq_dim_sharding = Shard(1)

    seq_parallel_sharding: PlacementList = [
        seq_dim_sharding,  # output
        seq_dim_sharding,  # logsumexp
        Replicate(),  # dropout_mask
        Replicate(),  # rng_state
        seq_dim_sharding,  # query
        seq_dim_sharding,  # key
        seq_dim_sharding,  # value
    ]
    if attn_mask_strategy is not None:
        seq_parallel_sharding.append(Replicate())

    single_mesh_dim_strategies.append(seq_parallel_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=4
    )


@register_op_strategy(
    torch.ops.aten._scaled_dot_product_attention_flash_musa_backward.default,
    schema_info=RuntimeSchemaInfo(8),
)
def scaled_dot_product_attention_flash_musa_backward_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    """DTensor sharding strategy of _scaled_dot_product_attention_flash_musa_backward"""
    # args: grad_out, query, key, value, output, logsumexp,
    # dropout_mask, rng_state, dropout_p, is_causal, attn_mask, scale

    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)

    args = op_schema.args_schema
    n_args = len(args)

    grad_out_strategy = args[0]  # grad_out
    query_strategy = args[1]  # query

    attn_mask_strategy = args[10] if n_args > 10 else None

    assert isinstance(grad_out_strategy, OpStrategy)
    assert isinstance(query_strategy, OpStrategy)

    query_shape = query_strategy.strategies[0].output_spec.shape
    is_4d = len(query_shape) == 4

    has_attn_mask = attn_mask_strategy is not None and isinstance(
        attn_mask_strategy, OpStrategy
    )

    single_mesh_dim_strategies = []

    all_replicate: PlacementList = [
        Replicate(),  # grad_query
        Replicate(),  # grad_key
        Replicate(),  # grad_value
        Replicate(),  # grad_attn_mask
        Replicate(),  # grad_out
        Replicate(),  # query
        Replicate(),  # key
        Replicate(),  # value
        Replicate(),  # output
        Replicate(),  # logsumexp
        Replicate(),  # dropout_mask
        Replicate(),  # rng_state
    ]
    if has_attn_mask:
        all_replicate.append(Replicate())  # attn_mask

    single_mesh_dim_strategies.append(all_replicate)

    if is_4d:
        # [batch_size, num_heads, seq_len, head_dim]
        tp_sharding = Shard(1)
    else:
        # [num_heads, seq_len, head_dim]
        tp_sharding = Shard(0)

    num_heads_dim_sharding: PlacementList = [
        tp_sharding,  # grad_query
        tp_sharding,  # grad_key
        tp_sharding,  # grad_value
        Replicate(),  # grad_attn_mask
        tp_sharding,  # grad_out
        tp_sharding,  # query
        tp_sharding,  # key
        tp_sharding,  # value
        tp_sharding,  # output
        tp_sharding,  # logsumexp
        Replicate(),  # dropout_mask
        Replicate(),  # rng_state
    ]
    if has_attn_mask:
        num_heads_dim_sharding.append(Replicate())  # attn_mask

    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Context Parallelism: shards on the sequence dim
    if is_4d:
        seq_dim_sharding = Shard(2)
    else:
        seq_dim_sharding = Shard(1)

    seq_parallel_sharding: PlacementList = [
        seq_dim_sharding,  # grad_query
        seq_dim_sharding,  # grad_key
        seq_dim_sharding,  # grad_value
        Replicate(),  # grad_attn_mask
        seq_dim_sharding,  # grad_out
        seq_dim_sharding,  # query
        seq_dim_sharding,  # key
        seq_dim_sharding,  # value
        seq_dim_sharding,  # output
        seq_dim_sharding,  # logsumexp
        Replicate(),  # dropout_mask
        Replicate(),  # rng_state
    ]
    if has_attn_mask:
        seq_parallel_sharding.append(Replicate())  # attn_mask

    single_mesh_dim_strategies.append(seq_parallel_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=4
    )
