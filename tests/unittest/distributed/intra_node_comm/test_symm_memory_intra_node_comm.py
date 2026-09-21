"""ProcessGroupMCCL IntraNodeComm tests with symmetric-memory tensors."""

# pylint: disable=protected-access,c-extension-no-member,wrong-import-order

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symmetric_memory

from test_intra_node_comm import (
    _assert_all_reduce_sum,
    _assert_flat_all_gather,
    _init_process_group,
    _reduce_scatter_expected,
    _run_spawned_test,
)

from torch_musa import testing


def _process_group_name():
    return dist._get_process_group_name(dist.group.WORLD)


def _symm_empty(shape, dtype, device):
    tensor = symmetric_memory.empty(shape, dtype=dtype, device=device)
    symm_mem = symmetric_memory.rendezvous(tensor, _process_group_name())
    symm_mem.barrier()
    return tensor


def _fill_reduce_scatter_input(tensor, rank, world_size):
    shard_numel = tensor.numel() // world_size
    for dst in range(world_size):
        begin = dst * shard_numel
        end = begin + shard_numel
        tensor[begin:end].fill_(rank * world_size + dst)


def _run_allgather(rank, world_size):
    counter = torch._C._distributed_c10d._get_mccl_intra_node_comm_usage_counter
    start_count = int(counter())
    device = torch.device(f"musa:{rank}")
    numel = 1024
    input_tensor = _symm_empty((numel,), torch.float32, device)
    input_tensor.fill_(rank)
    output_tensor = torch.empty(
        world_size * numel, dtype=input_tensor.dtype, device=device
    )

    work = dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True)
    assert work.wait()
    _assert_flat_all_gather(output_tensor, world_size)
    assert int(counter()) == start_count + 1


def _run_allreduce(rank, world_size):
    counter = torch._C._distributed_c10d._get_mccl_intra_node_comm_usage_counter
    start_count = int(counter())
    device = torch.device(f"musa:{rank}")
    numel = 1024
    input_tensor = _symm_empty((numel,), torch.float32, device)
    input_tensor.fill_(rank)

    work = dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM, async_op=True)
    assert work.wait()
    expected = world_size * (world_size - 1) // 2
    _assert_all_reduce_sum(input_tensor, expected)
    assert int(counter()) == start_count + 1


def _run_reduce_scatter(rank, world_size):
    counter = torch._C._distributed_c10d._get_mccl_intra_node_comm_usage_counter
    start_count = int(counter())
    device = torch.device(f"musa:{rank}")
    numel = 1024
    expected = _reduce_scatter_expected(rank, world_size)
    input_tensor = torch.empty(world_size * numel, dtype=torch.float32, device=device)
    _fill_reduce_scatter_input(input_tensor, rank, world_size)
    output_tensor = _symm_empty((numel,), input_tensor.dtype, device)
    work = dist.reduce_scatter_tensor(
        output_tensor, input_tensor, op=dist.ReduceOp.SUM, async_op=True
    )
    assert work.wait()
    _assert_all_reduce_sum(output_tensor, expected)
    assert int(counter()) == start_count + 1


def _worker(rank, world_size, rendezvous_file, test_case="reduce_scatter"):
    try:
        _init_process_group(rank, world_size, rendezvous_file)
        run_case = {
            "allgather": _run_allgather,
            "allreduce": _run_allreduce,
            "reduce_scatter": _run_reduce_scatter,
        }.get(test_case)
        if run_case is None:
            raise ValueError(f"Unknown test case: {test_case}")
        run_case(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
def test_pgmccl_symm_mem_intra_node_allgather():
    """Verify IntraNodeComm handles symmetric-memory allgather tensors."""
    _run_spawned_test(_worker, "allgather")


@testing.skip_if_not_multiple_musa_device
def test_pgmccl_symm_mem_intra_node_allreduce():
    """Verify IntraNodeComm handles symmetric-memory allreduce tensors."""
    _run_spawned_test(_worker, "allreduce")


@testing.skip_if_not_multiple_musa_device
def test_pgmccl_symm_mem_intra_node_reduce_scatter():
    """Verify IntraNodeComm handles symmetric-memory reduce_scatter tensors."""
    _run_spawned_test(_worker, "reduce_scatter")
