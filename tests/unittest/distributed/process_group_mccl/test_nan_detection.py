"""ProcessGroupMCCL NaN detection integration test."""

import os
import queue
import time
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_musa
from torch_musa import testing


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


def _nan_detection_worker(master_addr, master_port, status_queue):
    try:
        status_queue.put(("worker_started",))
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["TORCH_MCCL_NAN_CHECK"] = "1"
        os.environ["TORCH_MCCL_BLOCKING_WAIT"] = "1"

        torch_musa.set_device(0)
        status_queue.put(("device_set",))
        dist.init_process_group(
            "mccl",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=30),
        )
        status_queue.put(("pg_initialized",))

        tensor = torch.ones(256, dtype=torch.float32)
        tensor[0] = float("nan")
        tensor = tensor.musa()

        status_queue.put(("started_collective",))
        dist.all_reduce(tensor)
        torch.musa.synchronize()
        status_queue.put(("unexpected_success",))
    except Exception as err:  # pylint: disable=broad-except
        status_queue.put(("python_exception", repr(err), traceback.format_exc()))
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # pylint: disable=broad-except
            pass


@testing.skip_if_not_multiple_musa_device
def test_process_group_mccl_nan_detection_allreduce():
    """Verify NaN detection prevents all_reduce from completing normally."""
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < 1:
        pytest.skip("Need at least 1 MUSA device")

    master_addr, master_port = testing.gen_ip_port()
    ctx = mp.get_context("spawn")
    status_queue = ctx.Queue()
    proc = ctx.Process(
        target=_nan_detection_worker,
        args=(master_addr, master_port, status_queue),
    )

    proc.start()

    deadline = time.time() + 10
    while time.time() < deadline and proc.is_alive():
        time.sleep(0.2)

    exitcode = proc.exitcode
    statuses = _collect_statuses(status_queue)
    timed_out = proc.is_alive()

    if timed_out:
        _terminate_process(proc)
    else:
        pytest.fail(
            "nan detection worker exited unexpectedly before timeout. "
            f"exitcode={exitcode}, statuses={statuses}"
        )
