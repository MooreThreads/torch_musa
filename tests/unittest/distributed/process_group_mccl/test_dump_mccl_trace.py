"""Simple ProcessGroupMCCL trace pickle dump test."""

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


def _worker(rank, world_size, master_addr, master_port):
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"

        torch.musa.set_device(rank)
        dist.init_process_group(
            backend="mccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30),
        )

        device = torch.device(f"musa:{rank}")
        tensor = torch.ones(4, device=device) * (rank + 1)

        for _ in range(3):
            dist.all_reduce(tensor)
            torch.musa.synchronize(rank)

        dist.barrier()
        torch.musa.synchronize(rank)

        assert hasattr(torch._C._distributed_c10d, "_dump_mccl_trace")

        trace_bytes = torch._C._distributed_c10d._dump_mccl_trace(
            True,
            False,
            False,
        )
        trace_obj = pickle.loads(trace_bytes)

        assert "entries" in trace_obj
        assert len(trace_obj["entries"]) == 4
        assert trace_obj["entries"][-1]["profiling_name"] == "mccl:all_reduce_barrier"

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
def test_dump_mccl_trace_returns_valid_pickle():
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
