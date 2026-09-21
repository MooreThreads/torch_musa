"""ProcessGroupMCCL NaN detection integration test."""

import os
import queue
import signal
import time
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_musa
from torch_musa import testing


def _collect_statuses(status_queue):
    statuses = []
    while True:
        try:
            statuses.append(status_queue.get_nowait())
        except queue.Empty:
            break
    return statuses


def _is_nan_detection_error(err):
    text = f"{type(err).__name__}: {err}".lower()
    keywords = ("device-side assert",)
    return any(keyword in text for keyword in keywords)


def _init_nan_check_pg(master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["TORCH_MCCL_NAN_CHECK"] = "1"
    torch_musa.set_device(0)
    dist.init_process_group(
        "mccl",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )


def _put_status(status_queue, payload):
    status_queue.put(payload)
    status_queue.close()
    status_queue.join_thread()


def _nan_detection_worker(master_addr, master_port, status_queue):
    try:
        _init_nan_check_pg(master_addr, master_port)
        tensor = torch.ones(256, dtype=torch.float32)
        tensor[0] = float("nan")
        tensor = tensor.musa()
        dist.all_reduce(tensor)
        torch.musa.synchronize()
        _put_status(status_queue, ("unexpected_success", float(tensor[0].item())))
    except Exception as err:  # pylint: disable=broad-except
        _put_status(
            status_queue,
            (
                "nan_detected" if _is_nan_detection_error(err) else "python_exception",
                repr(err),
                traceback.format_exc(),
            ),
        )
        os._exit(0)


def _wait_for_status(status_queue, timeout_s):
    statuses = []
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            statuses.append(status_queue.get(timeout=0.2))
            return statuses
        except queue.Empty:
            continue
    return statuses


def _stop_process(proc):
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
    if proc.is_alive():
        proc.kill()
        proc.join(5)


def _run_nan_worker(timeout_s=30):
    master_addr, master_port = testing.gen_ip_port()
    ctx = mp.get_context("spawn")
    status_queue = ctx.Queue()
    proc = ctx.Process(
        target=_nan_detection_worker,
        args=(master_addr, master_port, status_queue),
    )
    proc.start()
    statuses = _wait_for_status(status_queue, timeout_s)
    _stop_process(proc)
    statuses.extend(_collect_statuses(status_queue))
    return proc.exitcode, statuses


def _is_abort_exit(exitcode):
    if exitcode in (-signal.SIGABRT, 134):
        return True
    return exitcode not in (0, None) and exitcode < 0


@testing.skip_if_not_multiple_musa_device
def test_process_group_mccl_nan_detection_allreduce():
    """Verify device-side NanCheck assert rejects all_reduce on NaN input."""
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.device_count() < 1:
        pytest.skip("Need at least 1 MUSA device")

    exitcode, statuses = _run_nan_worker()
    unexpected = [status for status in statuses if status[0] == "unexpected_success"]
    detected = [status for status in statuses if status[0] == "nan_detected"]
    other_errors = [status for status in statuses if status[0] == "python_exception"]

    assert not unexpected, (
        "all_reduce unexpectedly succeeded with NaN input. "
        f"exitcode={exitcode}, statuses={statuses}"
    )
    assert not other_errors, (
        "worker failed before NanCheck could run. "
        f"exitcode={exitcode}, statuses={statuses}"
    )
    assert detected or _is_abort_exit(exitcode), (
        "NanCheck assert(0) did not abort the worker or raise a device error. "
        f"exitcode={exitcode}, statuses={statuses}"
    )
