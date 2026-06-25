"""Simple ProcessGroupMCCL time estimator test."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_musa
from torch_musa import testing
from torch_musa.testing.common_utils import get_musa_arch

skip = torch.musa.mccl.version() < (2, 28, 9) or get_musa_arch() < 31
pytestmark = pytest.mark.skipif(skip, reason="Skip ProcessGroupMCCL tests")


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

        backend = dist.distributed_c10d._get_default_group()._get_backend(
            torch.device(f"musa:{rank}")
        )
        supports_time_estimate = bool(backend.supports_time_estimate)

        tensor = torch.ones(8, dtype=torch.float32, device=f"musa:{rank}")
        backend._start_time_estimate()
        if supports_time_estimate:
            dist.all_reduce(tensor)
            assert float(backend._end_time_estimate()) >= 0.0

        if supports_time_estimate:
            with dist._time_estimator(
                device=torch.device(f"musa:{rank}")
            ) as estimator_ctx:
                dist.all_reduce(tensor)
            assert float(estimator_ctx.estimated_time) >= 0.0

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
def test_process_group_mccl_time_estimator_behavior():
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
