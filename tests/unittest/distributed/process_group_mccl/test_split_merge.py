"""Simple ProcessGroupMCCL split/merge integration tests."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_musa  # pylint: disable=wrong-import-position
from torch_musa import testing  # pylint: disable=wrong-import-position
from torch_musa.testing.common_utils import get_musa_arch

skip = torch.musa.mccl.version() < (2, 28, 9) or get_musa_arch() < 31
pytestmark = pytest.mark.skipif(skip, reason="Skip ProcessGroupMCCL tests")


def _master_addr_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", str(sock.getsockname()[1])


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _split_worker(rank, world_size, master_addr, master_port):
    split_pg = None
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        torch_musa.set_device(rank)
        dist.init_process_group("mccl", rank=rank, world_size=world_size)

        parent_pg = dist.distributed_c10d._get_default_group()
        split_pg = parent_pg.split_group(
            [0],
            timeout=timedelta(seconds=30),
            group_name="mccl_split_rank0_only",
        )

        if rank == 0:
            assert split_pg is not None
            assert split_pg.size() == 1
            assert split_pg.rank() == 0
            tensor = torch.tensor([1.0], device=f"musa:{rank}")
            dist.all_reduce(tensor, group=split_pg)
            torch.musa.synchronize(rank)
            assert float(tensor.item()) == 1.0
        else:
            assert split_pg is None

    finally:
        try:
            if split_pg is not None:
                dist.destroy_process_group(split_pg)
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # pylint: disable=broad-except
            pass


def _merge_worker(rank, world_size, master_addr, master_port, merge_port):
    subgroup_0 = None
    subgroup_1 = None
    merged_pg = None
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        torch_musa.set_device(rank)
        dist.init_process_group("mccl", rank=rank, world_size=world_size)

        parent_pg = dist.distributed_c10d._get_default_group()
        subgroup_0 = parent_pg.split_group(
            [0],
            timeout=timedelta(seconds=30),
            group_name="mccl_merge_subgroup_0",
        )
        subgroup_1 = parent_pg.split_group(
            [1],
            timeout=timedelta(seconds=30),
            group_name="mccl_merge_subgroup_1",
        )
        local_subgroup = subgroup_0 if rank == 0 else subgroup_1
        assert local_subgroup is not None

        merge_store = dist.TCPStore(
            host_name=master_addr,
            port=int(merge_port),
            world_size=world_size,
            is_master=(rank == 0),
        )
        merged_pg = local_subgroup.merge_remote_group(
            merge_store,
            world_size,
            timedelta(seconds=40),
            "mccl_merged_simple",
        )
        assert merged_pg is not None
        assert merged_pg.size() == world_size

        tensor = torch.tensor([float(rank + 1)], device=f"musa:{rank}")
        dist.all_reduce(tensor, group=merged_pg)
        torch.musa.synchronize(rank)
        assert float(tensor.item()) == 3.0

    finally:
        for process_group in (merged_pg, subgroup_0, subgroup_1):
            try:
                if process_group is not None:
                    dist.destroy_process_group(process_group)
            except Exception:  # pylint: disable=broad-except
                pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # pylint: disable=broad-except
            pass


@testing.skip_if_not_multiple_musa_device
def test_process_group_mccl_split_simple():
    """Validate split() path with a minimal two-rank MCCL setup."""
    world_size = 2
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < world_size:
        pytest.skip("Need at least 2 MUSA devices")

    master_addr, master_port = _master_addr_port()
    mp.spawn(
        _split_worker,
        args=(
            world_size,
            master_addr,
            master_port,
        ),
        nprocs=world_size,
        join=True,
    )


@testing.skip_if_not_multiple_musa_device
def test_process_group_mccl_merge_simple():
    """Validate merge_remote_group() with two local singleton subgroups."""
    world_size = 2
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < world_size:
        pytest.skip("Need at least 2 MUSA devices")

    master_addr, master_port = _master_addr_port()
    merge_port = _free_port()
    mp.spawn(
        _merge_worker,
        args=(
            world_size,
            master_addr,
            master_port,
            merge_port,
        ),
        nprocs=world_size,
        join=True,
    )
