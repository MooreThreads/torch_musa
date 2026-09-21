"""ProcessGroupMCCL IntraNodeComm coalesced integration tests."""

# pylint: disable=protected-access,c-extension-no-member,no-member,too-many-locals

import torch
import torch.distributed as dist

from test_intra_node_comm import (
    _assert_all_reduce_sum,
    _assert_flat_all_gather,
    _init_process_group,
    _make_reduce_scatter_chunks,
    _reduce_scatter_expected,
    _run_spawned_test,
)

from torch_musa import testing


def _run_coalesced(rank, world_size):
    counter = torch._C._distributed_c10d._get_mccl_intra_node_comm_usage_counter
    start_count = int(counter())
    expected = world_size * (world_size - 1) // 2
    device = torch.device(f"musa:{rank}")
    numels = (1024, 2048)

    allreduce_tensors = [
        torch.full((numel,), rank, dtype=torch.float32, device=device)
        for numel in numels
    ]
    allreduce_opts = dist.AllreduceCoalescedOptions()
    allreduce_opts.reduceOp = dist.ReduceOp.SUM
    allreduce_opts.asyncOp = True
    work = dist.group.WORLD.allreduce_coalesced(allreduce_tensors, allreduce_opts)
    assert work.wait()
    for tensor in allreduce_tensors:
        _assert_all_reduce_sum(tensor, expected)
    assert int(counter()) == start_count + len(numels)

    allgather_inputs = [
        torch.full((numel,), rank, dtype=torch.float32, device=device)
        for numel in numels
    ]
    allgather_outputs = [
        torch.empty(world_size * tensor.numel(), dtype=tensor.dtype, device=device)
        for tensor in allgather_inputs
    ]
    allgather_opts = dist.distributed_c10d.AllgatherOptions()
    allgather_opts.asyncOp = True
    work = dist.group.WORLD.allgather_into_tensor_coalesced(
        allgather_outputs, allgather_inputs, allgather_opts
    )
    assert work.wait()
    for output in allgather_outputs:
        _assert_flat_all_gather(output, world_size)
    assert int(counter()) == start_count + 2 * len(numels)

    reduce_scatter_inputs = []
    reduce_scatter_outputs = []
    for numel in numels:
        reduce_scatter_inputs.append(
            torch.cat(_make_reduce_scatter_chunks(rank, world_size, numel, device))
        )
        reduce_scatter_outputs.append(
            torch.empty(numel, dtype=torch.float32, device=device)
        )

    reduce_scatter_opts = dist.ReduceScatterOptions()
    reduce_scatter_opts.reduceOp = dist.ReduceOp.SUM
    reduce_scatter_opts.asyncOp = True
    work = dist.group.WORLD.reduce_scatter_tensor_coalesced(
        reduce_scatter_outputs, reduce_scatter_inputs, reduce_scatter_opts
    )
    assert work.wait()
    reduce_scatter_expected = _reduce_scatter_expected(rank, world_size)
    for output in reduce_scatter_outputs:
        _assert_all_reduce_sum(output, reduce_scatter_expected)
    assert int(counter()) == start_count + 3 * len(numels)


def _worker(rank, world_size, rendezvous_file):
    try:
        _init_process_group(rank, world_size, rendezvous_file)
        assert torch.musa.current_device() == rank
        _run_coalesced(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
def test_pgmccl_intra_node_comm_coalesced():
    """Verify ProcessGroupMCCL uses IntraNodeComm for local coalesced ops."""
    _run_spawned_test(_worker)
