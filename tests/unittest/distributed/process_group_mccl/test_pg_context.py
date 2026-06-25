"""Simple ProcessGroupMCCL dist2 process_group context test."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_musa
from torch_musa import testing

dist2 = pytest.importorskip("torch.distributed._dist2")


def _master_addr_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", str(sock.getsockname()[1])


def _worker(rank, world_size, master_addr, master_port):
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["TORCH_MCCL_BLOCKING_WAIT"] = "1"

        torch_musa.set_device(rank)
        dist.init_process_group("mccl", rank=rank, world_size=world_size)

        default_process_group = dist.distributed_c10d._get_default_group()
        prev_pg = dist2.current_process_group()
        tensor = torch.tensor(
            [float(rank + 1)],
            dtype=torch.float32,
            device=f"musa:{rank}",
        )

        with dist2.process_group(default_process_group):
            current = dist2.current_process_group()
            assert current is default_process_group
            current.allreduce(tensor).wait()

        torch.musa.synchronize(rank)
        assert float(tensor.item()) == 3.0
        assert dist2.current_process_group() is prev_pg
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
def test_process_group_mccl_dist2_context_allreduce():
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
