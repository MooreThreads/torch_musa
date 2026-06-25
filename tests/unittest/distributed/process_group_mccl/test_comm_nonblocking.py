"""Simple ProcessGroupMCCL non-blocking communicator init test."""

import os
import queue
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_musa import testing
from torch_musa.testing.common_utils import get_musa_arch

skip = torch.musa.mccl.version() < (2, 28, 9) or get_musa_arch() < 31
pytestmark = pytest.mark.skipif(skip, reason="Skip ProcessGroupMCCL tests")

_COMM_TIMEOUT_SECONDS = 10
_PROCESS_EXIT_TIMEOUT_SECONDS = 60
_HANG_SECONDS = 1000000


def _master_addr_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return "127.0.0.1", str(sock.getsockname()[1])


def _set_nonblocking_mccl_env(master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["TORCH_MCCL_USE_COMM_NONBLOCKING"] = "1"
    os.environ["TORCH_MCCL_NONBLOCKING_TIMEOUT"] = str(_COMM_TIMEOUT_SECONDS)


def _terminate_process(proc):
    if proc.is_alive():
        proc.terminate()
    proc.join(10)
    if proc.is_alive():
        proc.kill()
    proc.join(5)


def _collect_statuses(status_queue):
    statuses = []
    while True:
        try:
            statuses.append(status_queue.get_nowait())
        except queue.Empty:
            break
    return statuses


def _worker(rank, world_size, master_addr, master_port, status_queue):
    try:
        _set_nonblocking_mccl_env(master_addr, master_port)
        torch.musa.set_device(rank)
        dist.init_process_group(
            backend="mccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=_COMM_TIMEOUT_SECONDS),
        )

        if rank == 1:
            status_queue.put((rank, "sleeping_before_collective"))
            time.sleep(_HANG_SECONDS)
            status_queue.put((rank, "unexpected_wakeup"))
            return

        tensor = torch.randn(10, 10, device=torch.device("musa", rank))
        status_queue.put((rank, "starting_all_reduce"))
        dist.all_reduce(tensor, async_op=False)
        torch.musa.synchronize(rank)
        status_queue.put((rank, "unexpected_success"))
    except Exception as err:  # pylint: disable=broad-except
        status_queue.put((rank, "exception", repr(err)))
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as err:  # pylint: disable=broad-except
            status_queue.put((rank, "destroy_process_group_exception", repr(err)))
        status_queue.put((rank, "exiting"))


@testing.skip_if_not_multiple_musa_device
def test_comm_nonblocking_exits_if_peer_hangs():
    """Rank 0 must not hang forever if rank 1 never initializes the collective."""
    world_size = 2
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < world_size:
        pytest.skip("Need at least 2 MUSA devices")

    master_addr, master_port = _master_addr_port()
    ctx = mp.get_context("spawn")
    status_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker,
            args=(rank, world_size, master_addr, master_port, status_queue),
        )
        for rank in range(world_size)
    ]

    for proc in procs:
        proc.start()

    rank0_proc = procs[0]
    deadline = time.time() + _PROCESS_EXIT_TIMEOUT_SECONDS
    while rank0_proc.is_alive() and time.time() < deadline:
        time.sleep(0.2)

    rank0_timed_out = rank0_proc.is_alive()
    for proc in procs:
        _terminate_process(proc)

    statuses = _collect_statuses(status_queue)
    rank0_statuses = [status for status in statuses if status[0] == 0]
    rank1_statuses = [status for status in statuses if status[0] == 1]

    assert not rank0_timed_out, (
        "rank 0 did not exit while rank 1 skipped the collective. "
        f"statuses={statuses}"
    )
    assert rank0_proc.exitcode == 0, (
        f"rank 0 exited abnormally with exitcode={rank0_proc.exitcode}. "
        f"statuses={statuses}"
    )
    assert any(status[1] == "exception" for status in rank0_statuses), (
        "rank 0 should report the MCCL non-blocking timeout/error instead of "
        f"hanging or succeeding. statuses={statuses}"
    )
    assert not any(
        status[1] == "unexpected_success" for status in rank0_statuses
    ), f"rank 0 unexpectedly completed all_reduce. statuses={statuses}"
    assert any(
        status[1] == "sleeping_before_collective" for status in rank1_statuses
    ), (
        "rank 1 should initialize the process group and then hang before all_reduce. "
        f"statuses={statuses}"
    )
