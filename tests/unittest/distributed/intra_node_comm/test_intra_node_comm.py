"""ProcessGroupMCCL IntraNodeComm integration tests."""

# pylint: disable=protected-access,c-extension-no-member,too-many-locals

import ctypes
import os
import tempfile
from ctypes.util import find_library
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_musa import testing


def _assert_all_reduce_sum(tensor, expected):
    assert bool(tensor.eq(expected).all().item())


def _assert_all_gather(output_tensors, world_size, value_offset=0):
    assert len(output_tensors) == world_size
    for rank, tensor in enumerate(output_tensors):
        assert bool(tensor.eq(rank + value_offset).all().item())


def _assert_flat_all_gather(output_tensor, world_size, value_offset=0):
    chunks = output_tensor.view(world_size, -1)
    for rank, tensor in enumerate(chunks):
        assert bool(tensor.eq(rank + value_offset).all().item())


def _reduce_scatter_expected(rank, world_size):
    return sum(peer * world_size + rank for peer in range(world_size))


def _make_reduce_scatter_chunks(rank, world_size, numel, device):
    return [
        torch.full(
            (numel,),
            rank * world_size + dst,
            dtype=torch.float32,
            device=device,
        )
        for dst in range(world_size)
    ]


def _mtlink_port_count(src, dst):
    libmusart = find_library("musart") or "libmusart.so"
    try:
        musart = ctypes.CDLL(libmusart)
    except OSError:
        pytest.skip("libmusart is not available for MTLink topology query")

    mtlink_attr_port_count = 5
    value = ctypes.c_int(0)
    func = musart.musaDeviceGetP2PAttribute
    func.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    func.restype = ctypes.c_int
    ret = func(
        ctypes.byref(value),
        mtlink_attr_port_count,
        src,
        dst,
    )
    if ret != 0:
        return 0
    return value.value


def _init_process_group(rank, world_size, rendezvous_file):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["TORCH_MCCL_BLOCKING_WAIT"] = "0"
    os.environ["ENABLE_INTRA_NODE_COMM"] = "1"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"  # TORCH_CPP_LOG_LEVEL=INFO

    dist.init_process_group(
        backend="mccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )


def _run_allreduce(rank, world_size):
    counter = torch._C._distributed_c10d._get_mccl_intra_node_comm_usage_counter
    start_count = int(counter())
    expected = world_size * (world_size - 1) // 2
    device = torch.device(f"musa:{rank}")

    tensors = [
        torch.full((numel,), rank, dtype=torch.float32, device=device)
        for numel in (1024, 512 * 1024 // 4)
    ]
    works = [
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        for tensor in tensors
    ]

    # Sync work must be ordered after outstanding async work that uses the same
    # internal symmetric-memory workspace.
    sync_value_offset = 10
    sync_tensor = torch.full(
        (1024,), rank + sync_value_offset, dtype=torch.float32, device=device
    )
    dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)

    for tensor, work in zip(tensors, works):
        assert work.wait()
        _assert_all_reduce_sum(tensor, expected)

    _assert_all_reduce_sum(sync_tensor, expected + sync_value_offset * world_size)
    assert int(counter()) == start_count + len(tensors) + 1


def _run_allgather(rank, world_size):
    counter = torch._C._distributed_c10d._get_mccl_intra_node_comm_usage_counter
    start_count = int(counter())
    device = torch.device(f"musa:{rank}")
    input_tensor = torch.full((1024,), rank, dtype=torch.float32, device=device)
    large_numel = 10 * 1024 * 1024 // 4 + 1

    flat_output = torch.empty(
        world_size * input_tensor.numel(),
        dtype=input_tensor.dtype,
        device=device,
    )
    work = dist.all_gather_into_tensor(flat_output, input_tensor, async_op=True)

    output_tensors = [torch.empty_like(input_tensor) for _ in range(world_size)]
    list_work = dist.all_gather(output_tensors, input_tensor, async_op=True)

    sync_value_offset = 10
    sync_input_tensor = torch.full_like(input_tensor, rank + sync_value_offset)
    sync_flat_output = torch.empty_like(flat_output)
    dist.all_gather_into_tensor(sync_flat_output, sync_input_tensor)

    assert work.wait()
    assert list_work.wait()
    _assert_flat_all_gather(flat_output, world_size)
    _assert_all_gather(output_tensors, world_size)
    _assert_flat_all_gather(sync_flat_output, world_size, sync_value_offset)
    assert int(counter()) == start_count + 3

    large_input_tensor = torch.full(
        (large_numel,), rank, dtype=torch.float32, device=device
    )
    large_flat_output = torch.empty(
        world_size * large_input_tensor.numel(),
        dtype=large_input_tensor.dtype,
        device=device,
    )
    work = dist.all_gather_into_tensor(
        large_flat_output, large_input_tensor, async_op=True
    )
    assert work.wait()
    _assert_flat_all_gather(large_flat_output, world_size)
    assert int(counter()) == start_count + 4


def _run_reduce_scatter(rank, world_size):
    counter = torch._C._distributed_c10d._get_mccl_intra_node_comm_usage_counter
    start_count = int(counter())
    device = torch.device(f"musa:{rank}")
    numel = 1024
    large_numel = 10 * 1024 * 1024 // 4 + 1
    expected = _reduce_scatter_expected(rank, world_size)

    chunks = _make_reduce_scatter_chunks(rank, world_size, numel, device)
    input_tensor = torch.cat(chunks)
    output_tensor = torch.empty(numel, dtype=input_tensor.dtype, device=device)
    work = dist.reduce_scatter_tensor(
        output_tensor, input_tensor, op=dist.ReduceOp.SUM, async_op=True
    )

    input_tensors = _make_reduce_scatter_chunks(rank, world_size, numel, device)
    list_output_tensor = torch.empty_like(input_tensors[rank])
    list_work = dist.reduce_scatter(
        list_output_tensor, input_tensors, op=dist.ReduceOp.SUM, async_op=True
    )

    sync_value_offset = 10
    sync_chunks = [
        tensor + sync_value_offset
        for tensor in _make_reduce_scatter_chunks(rank, world_size, numel, device)
    ]
    sync_input_tensor = torch.cat(sync_chunks)
    sync_output_tensor = torch.empty_like(output_tensor)
    dist.reduce_scatter_tensor(
        sync_output_tensor, sync_input_tensor, op=dist.ReduceOp.SUM
    )

    assert work.wait()
    _assert_all_reduce_sum(output_tensor, expected)
    assert list_work.wait()
    _assert_all_reduce_sum(list_output_tensor, expected)
    _assert_all_reduce_sum(
        sync_output_tensor, expected + sync_value_offset * world_size
    )
    assert int(counter()) == start_count + 3

    large_chunks = _make_reduce_scatter_chunks(rank, world_size, large_numel, device)
    large_input_tensor = torch.cat(large_chunks)
    large_output_tensor = torch.empty(
        large_numel, dtype=large_input_tensor.dtype, device=device
    )
    work = dist.reduce_scatter_tensor(
        large_output_tensor,
        large_input_tensor,
        op=dist.ReduceOp.SUM,
        async_op=True,
    )
    assert work.wait()
    _assert_all_reduce_sum(large_output_tensor, expected)
    assert int(counter()) == start_count + 4


CASE_RUNNERS = {
    "allreduce": _run_allreduce,
    "allgather": _run_allgather,
    "reduce_scatter": _run_reduce_scatter,
}


def _worker(rank, world_size, rendezvous_file, test_case):
    try:
        _init_process_group(rank, world_size, rendezvous_file)
        assert torch.musa.current_device() == rank
        run_case = CASE_RUNNERS.get(test_case)
        if run_case is None:
            raise ValueError(f"Unknown test case: {test_case}")
        run_case(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned_test(worker, *worker_args):
    world_size = 2
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < world_size:
        pytest.skip("Need at least 2 MUSA devices")

    for src in range(world_size):
        for dst in range(world_size):
            if src != dst and _mtlink_port_count(src, dst) <= 0:
                pytest.skip("MUSA IntraNodeComm requires MTLink access")

    with tempfile.TemporaryDirectory() as tmpdir:
        rendezvous_file = os.path.join(tmpdir, "pgmccl_intra_node_comm")
        mp.spawn(
            worker,
            args=(world_size, rendezvous_file, *worker_args),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize(
    "test_case",
    ("allreduce", "allgather", "reduce_scatter"),
)
@testing.skip_if_not_multiple_musa_device
def test_pgmccl_intra_node_comm(test_case):
    """Verify ProcessGroupMCCL uses IntraNodeComm for local collectives."""
    _run_spawned_test(_worker, test_case)
