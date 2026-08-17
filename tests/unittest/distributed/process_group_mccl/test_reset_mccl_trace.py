"""ProcessGroupMCCL Flight Recorder reset test."""

import os
import pickle
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_musa import testing  # pylint: disable=wrong-import-position


def _master_addr_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", str(sock.getsockname()[1])


def _dump_entries():
    trace_bytes = torch._C._distributed_c10d._dump_mccl_trace(
        True,
        False,
        False,
    )
    return pickle.loads(trace_bytes)["entries"]


def _run_all_reduce(rank, count):
    device = torch.device(f"musa:{rank}")
    tensor = torch.ones(4, device=device) * (rank + 1)
    for _ in range(count):
        dist.all_reduce(tensor)
    torch.musa.synchronize(rank)


def _worker(rank, world_size, master_addr, master_port):
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["TORCH_MCCL_TRACE_BUFFER_SIZE"] = "2000"

        torch.musa.set_device(rank)
        dist.init_process_group(
            backend="mccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30),
        )

        assert hasattr(torch._C._distributed_c10d, "_dump_mccl_trace")
        assert hasattr(torch._C._distributed_c10d, "_reset_fr_recording_mccl")

        _run_all_reduce(rank, 5)
        assert len(_dump_entries()) == 5

        torch._C._distributed_c10d._reset_fr_recording_mccl()

        _run_all_reduce(rank, 3)
        entries = _dump_entries()
        assert len(entries) == 3
        assert [entry["record_id"] for entry in entries] == [0, 1, 2]
        assert all(entry["profiling_name"] == "mccl:all_reduce" for entry in entries)

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
def test_reset_mccl_trace_discards_old_entries():
    world_size = 2
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < world_size:
        pytest.skip("Need at least 2 MUSA devices")

    master_addr, master_port = _master_addr_port()
    mp.spawn(
        _worker,
        args=(world_size, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )
