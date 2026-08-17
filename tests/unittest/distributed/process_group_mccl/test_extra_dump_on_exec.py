"""ProcessGroupMCCL extra dump-on-exec test."""

import os
import pickle
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
import torch.multiprocessing as mp

from torch_musa import testing  # pylint: disable=wrong-import-position


def _master_addr_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", str(sock.getsockname()[1])


def _worker(rank, world_size, master_addr, master_port, dump_prefix):
    destroyed = False
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["TORCH_MCCL_TRACE_BUFFER_SIZE"] = "2000"
        os.environ["TORCH_MCCL_EXTRA_DUMP_ON_EXEC"] = "1"
        os.environ["TORCH_FR_DUMP_TEMP_FILE"] = dump_prefix

        torch.musa.set_device(rank)
        dist.init_process_group(
            backend="mccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30),
        )

        device = torch.device(f"musa:{rank}")
        tensor = torch.ones(4, device=device) * (rank + 1)
        dist.all_reduce(tensor)
        torch.musa.synchronize(rank)

        dump_file = f"{dump_prefix}{rank}"
        assert not os.path.exists(dump_file)

        c10d._abort_process_group()  # pylint: disable=protected-access
        destroyed = True

        assert os.path.exists(dump_file)
        with open(dump_file, "rb") as trace_file:
            trace_obj = pickle.load(trace_file)
            print(trace_obj)

        assert "entries" in trace_obj
        assert any(
            entry["profiling_name"] == "mccl:all_reduce"
            for entry in trace_obj["entries"]
        )
    finally:
        if not destroyed and dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
def test_extra_dump_on_exec_writes_trace(tmp_path):
    """Test extra dump-on-exec writes flight recorder trace."""
    world_size = 2
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < world_size:
        pytest.skip("Need at least 2 MUSA devices")

    master_addr, master_port = _master_addr_port()
    dump_prefix = os.path.join(str(tmp_path), "extra_dump_rank_")
    mp.spawn(
        _worker,
        args=(world_size, master_addr, master_port, dump_prefix),
        nprocs=world_size,
        join=True,
    )
