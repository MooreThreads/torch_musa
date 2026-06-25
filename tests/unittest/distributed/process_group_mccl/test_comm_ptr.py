"""Simple ProcessGroupMCCL communicator pointer test."""

import ctypes
import os
import socket
from ctypes.util import find_library
from datetime import timedelta

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


def _load_mccl():
    libmccl = find_library("mccl")
    if libmccl is None:
        candidates = ("libmccl.so", "libmccl.so.1", "libmccl.so.2")
        for candidate in candidates:
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                continue
        raise RuntimeError("Unable to load MCCL shared library")
    return ctypes.CDLL(libmccl)


def _worker(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch_musa.set_device(rank)

    dist.init_process_group(
        backend="mccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )

    try:
        # Trigger one collective first so the communicator is created.
        tensor = torch.ones(1, device=f"musa:{rank}")
        dist.all_reduce(tensor)
        torch.musa.synchronize(rank)

        backend = dist.group.WORLD._get_backend(torch.device("musa"))
        comm_ptr = backend._comm_ptr()

        print(f"[rank{rank}] comm_ptr = {comm_ptr:#x}")

        assert isinstance(comm_ptr, int)
        assert comm_ptr != 0
        assert dist.get_world_size() == world_size

        mccl = _load_mccl()
        mccl.mcclCommUserRank.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        mccl.mcclCommUserRank.restype = ctypes.c_int

        mccl.mcclCommCount.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        mccl.mcclCommCount.restype = ctypes.c_int

        mccl.mcclGetErrorString.argtypes = [ctypes.c_int]
        mccl.mcclGetErrorString.restype = ctypes.c_char_p

        rank_from_pg = dist.get_rank()
        world_size_from_pg = dist.get_world_size()
        rank_from_comm = ctypes.c_int(-1)
        world_size_from_comm = ctypes.c_int(-1)
        ret = mccl.mcclCommUserRank(
            ctypes.c_void_p(comm_ptr),
            ctypes.byref(rank_from_comm),
        )

        if ret != 0:
            err = mccl.mcclGetErrorString(ret).decode()
            raise RuntimeError(f"mcclCommUserRank failed: ret={ret}, err={err}")

        ret = mccl.mcclCommCount(
            ctypes.c_void_p(comm_ptr),
            ctypes.byref(world_size_from_comm),
        )

        if ret != 0:
            err = mccl.mcclGetErrorString(ret).decode()
            raise RuntimeError(f"mcclCommCount failed: ret={ret}, err={err}")

        assert rank_from_comm.value == rank_from_pg
        assert world_size_from_comm.value == world_size_from_pg
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
def test_comm_ptr_world_size():
    """Validate MCCL communicator pointer metadata for a two-rank world."""
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
