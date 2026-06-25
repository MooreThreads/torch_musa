"""ProcessGroupMCCL contiguity check integration tests."""

import os
import queue
import time
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_musa  # pylint: disable=wrong-import-position
from torch_musa import testing  # pylint: disable=wrong-import-position


def _terminate_processes(processes):
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(10)
    for proc in processes:
        if proc.is_alive():
            proc.kill()
    for proc in processes:
        proc.join(5)


def _collect_statuses(status_queue):
    statuses = []
    while True:
        try:
            statuses.append(status_queue.get_nowait())
        except queue.Empty:
            break
    return statuses


def _run_workers(worker, world_size, timeout_s=90):
    master_addr, master_port = testing.gen_ip_port()
    ctx = mp.get_context("spawn")
    status_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=worker,
            args=(rank, world_size, master_addr, master_port, status_queue),
        )
        for rank in range(world_size)
    ]

    for proc in processes:
        proc.start()

    deadline = time.time() + timeout_s
    while time.time() < deadline and any(proc.is_alive() for proc in processes):
        time.sleep(0.2)

    statuses = _collect_statuses(status_queue)
    timed_out = any(proc.is_alive() for proc in processes)
    _terminate_processes(processes)
    return statuses, timed_out


def _p2p_noncontiguous_worker(rank, world_size, master_addr, master_port, status_queue):
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["TORCH_MCCL_BLOCKING_WAIT"] = "1"

        if world_size != 2:
            raise ValueError(f"Expected world_size=2, got {world_size}")

        torch_musa.set_device(rank)
        dist.init_process_group("mccl", rank=rank, world_size=world_size)

        if rank == 0:
            tensor = (
                (torch.arange(6, dtype=torch.float32, device=f"musa:{rank}") + 42)
                .view(2, 3)
                .t()
            )
            assert not tensor.is_contiguous()
            dist.send(tensor, dst=1)
        else:
            tensor = torch.empty((2, 3), dtype=torch.float32, device=f"musa:{rank}").t()
            assert not tensor.is_contiguous()
            dist.recv(tensor, src=0)

            expected = (
                (torch.arange(6, dtype=torch.float32, device=f"musa:{rank}") + 42)
                .view(2, 3)
                .t()
            )
            torch.testing.assert_close(tensor, expected)

        dist.barrier()
        torch.musa.synchronize(rank)
        status_queue.put(("ok", rank))
    except Exception as err:  # pylint: disable=broad-except
        status_queue.put(("error", rank, repr(err), traceback.format_exc()))
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # pylint: disable=broad-except
            pass


def _collective_noncontiguous_worker(
    rank, world_size, master_addr, master_port, status_queue
):
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["TORCH_MCCL_BLOCKING_WAIT"] = "1"

        torch_musa.set_device(rank)
        dist.init_process_group("mccl", rank=rank, world_size=world_size)

        tensor = (
            torch.arange(16, dtype=torch.float32, device=f"musa:{rank}").view(4, 4).t()
        )
        assert not tensor.is_contiguous()

        try:
            dist.all_reduce(tensor)
            torch.musa.synchronize(rank)
            status_queue.put(("unexpected_success", rank))
        except Exception as err:  # pylint: disable=broad-except
            status_queue.put(("noncontiguous_rejected", rank, repr(err)))
    except Exception as err:  # pylint: disable=broad-except
        status_queue.put(("error", rank, repr(err), traceback.format_exc()))
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # pylint: disable=broad-except
            pass


@testing.skip_if_not_multiple_musa_device
def test_process_group_mccl_p2p_noncontiguous_tensor_allowed():  # pylint: disable=invalid-name
    """Verify non-contiguous tensors are accepted for MCCL P2P send/recv."""
    world_size = 2
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < world_size:
        pytest.skip("Need at least 2 MUSA devices")

    statuses, timed_out = _run_workers(_p2p_noncontiguous_worker, world_size)
    assert not timed_out, f"p2p worker timed out. statuses={statuses}"

    errors = [status for status in statuses if status[0] == "error"]
    assert not errors, f"p2p worker failed: {errors}"

    ok_ranks = sorted(status[1] for status in statuses if status[0] == "ok")
    assert ok_ranks == [0, 1], f"p2p worker incomplete. statuses={statuses}"


@testing.skip_if_not_multiple_musa_device
def test_process_group_mccl_collective_noncontiguous_tensor_rejected():  # pylint: disable=invalid-name
    """Verify collective all_reduce rejects non-contiguous tensors."""
    world_size = 2
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < world_size:
        pytest.skip("Need at least 2 MUSA devices")

    statuses, timed_out = _run_workers(_collective_noncontiguous_worker, world_size)
    assert not timed_out, f"collective worker timed out. statuses={statuses}"

    errors = [status for status in statuses if status[0] == "error"]
    assert not errors, f"collective worker failed: {errors}"

    success_ranks = sorted(
        status[1] for status in statuses if status[0] == "unexpected_success"
    )
    assert not success_ranks, (
        "non-contiguous all_reduce unexpectedly succeeded on ranks: "
        f"{success_ranks}. statuses={statuses}"
    )

    rejected = [status for status in statuses if status[0] == "noncontiguous_rejected"]
    rejected_ranks = sorted(status[1] for status in rejected)
    assert rejected_ranks == [0, 1], (
        "non-contiguous all_reduce rejection was not observed on all ranks. "
        f"statuses={statuses}"
    )

    for _, rank, err_repr in rejected:
        assert (
            "contiguous" in err_repr.lower()
        ), f"rank {rank} error does not mention contiguous requirement: {err_repr}"
