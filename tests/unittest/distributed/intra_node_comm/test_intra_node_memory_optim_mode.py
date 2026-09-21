"""Tests for the IntraNodeComm reduce-scatter memory optimization mode."""

# pylint: disable=protected-access,c-extension-no-member,wrong-import-order

import os

import torch
import torch.distributed as dist

from test_intra_node_comm import (
    _assert_all_reduce_sum,
    _init_process_group,
    _make_reduce_scatter_chunks,
    _reduce_scatter_expected,
    _run_spawned_test,
)

from torch_musa import testing


def _run_reduce_scatter(rank, world_size):
    counter = torch._C._distributed_c10d._get_mccl_intra_node_comm_usage_counter
    start_count = int(counter())
    device = torch.device(f"musa:{rank}")
    numel = 1024
    expected = _reduce_scatter_expected(rank, world_size)

    input_tensor = torch.cat(
        _make_reduce_scatter_chunks(rank, world_size, numel, device)
    )
    output_tensor = torch.empty(numel, dtype=input_tensor.dtype, device=device)
    work = dist.reduce_scatter_tensor(
        output_tensor, input_tensor, op=dist.ReduceOp.SUM, async_op=True
    )

    assert work.wait()
    _assert_all_reduce_sum(output_tensor, expected)
    assert int(counter()) == start_count + 1


def _worker(rank, world_size, rendezvous_file):
    try:
        os.environ["INTRA_NODE_MEMORY_OPTIM_MODE"] = "1"
        _init_process_group(rank, world_size, rendezvous_file)
        _run_reduce_scatter(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
def test_pgmccl_intra_node_memory_optim_mode():
    """Verify output staging works for reduce-scatter in memory optim mode."""
    _run_spawned_test(_worker)
