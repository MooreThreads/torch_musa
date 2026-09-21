"""Tests for public torch.musa.ipc APIs."""

# pylint: disable=missing-function-docstring,protected-access

import json
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch_musa import testing


def _producer_tensor_handle(handle_q, done_q):
    tensor = torch.arange(16, device="musa", dtype=torch.float32).reshape(4, 4)
    handle_q.put(torch.musa.ipc.export_tensor(tensor))
    done_q.get()


def _consume_tensor_handle(handle):
    opened = torch.musa.ipc.open_tensor(handle)
    opened.close()


def _assert_handle_opens_from_spawned_process(handle):
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=_consume_tensor_handle, args=(handle,))
    process.start()
    process.join(30)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("MUSA IPC consumer process timed out")

    assert process.exitcode == 0


def test_export_tensor_rejects_non_tensor():
    with pytest.raises(TypeError, match="torch.Tensor"):
        torch.musa.ipc.export_tensor(object())


@testing.skip_if_musa_unavailable
def test_export_tensor_rejects_cpu_tensor():
    tensor = torch.empty(2)
    with pytest.raises(RuntimeError, match="MUSA"):
        torch.musa.ipc.export_tensor(tensor)


@testing.skip_if_musa_unavailable
def test_open_tensor_wraps_owner_and_close(monkeypatch):
    handle = SimpleNamespace(
        dtype=torch.float16,
        storage_offset=0,
        size=(0,),
        stride=(1,),
        requires_grad=False,
    )

    fake_storage = torch.empty(0).untyped_storage()
    monkeypatch.setattr(torch.musa.ipc, "_open_storage", lambda _handle: fake_storage)

    owner = torch.musa.ipc.open_tensor(handle)

    assert owner.closed is False
    assert owner.tensor.device.type == "cpu"
    owner.close()
    assert owner.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        _ = owner.tensor


def test_tensor_owner_context_manager_closes_owner():
    owner = torch.musa.ipc.MUSAIPCTensor(torch.empty(0))

    with owner as entered:
        assert entered is owner
        assert owner.closed is False

    assert owner.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        with owner:
            pass


@testing.skip_if_musa_unavailable
def test_exported_handle_to_dict_from_dict_roundtrip():
    base = torch.arange(64, device="musa", dtype=torch.float16).reshape(8, 8)
    tensor = base[2:6, 1:7:2]
    handle = torch.musa.ipc.export_tensor(tensor)

    data = handle.to_dict()
    json_data = json.loads(json.dumps(data))
    assert json_data == data
    assert data["dtype"] == "torch.float16"
    restored = torch.musa.ipc.MUSAIPCTensorHandle.from_dict(json_data)
    assert restored == handle

    _assert_handle_opens_from_spawned_process(restored)


@testing.skip_if_musa_unavailable
def test_export_tensor_handle_preserves_view_metadata():
    base = torch.arange(64, device="musa", dtype=torch.float16).reshape(8, 8)
    view = base[2:6, 1:7:2]

    handle = torch.musa.ipc.export_tensor(view)

    assert handle.dtype is torch.float16
    assert handle.size == tuple(view.size())
    assert handle.stride == tuple(view.stride())
    assert handle.storage_offset == view.storage_offset()
    assert handle.requires_grad is False
    assert handle.device_uuid == str(torch.musa.get_device_properties(view.device).uuid)

    assert handle.shared_storage_info[0] == 0
    assert len(handle.shared_storage_info) == 8

    _assert_handle_opens_from_spawned_process(handle)


@testing.skip_if_musa_unavailable
def test_open_tensor_from_spawned_process_handle():
    ctx = mp.get_context("spawn")
    handle_q = ctx.Queue()
    done_q = ctx.Queue()
    process = ctx.Process(target=_producer_tensor_handle, args=(handle_q, done_q))
    process.start()

    try:
        handle = handle_q.get(timeout=30)
        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            opened = torch.musa.ipc.open_tensor(handle)
            assert opened.tensor.device.type == "musa"
            copied = opened.tensor.clone()
            opened.close()

        assert copied.tolist() == [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
        ]
    finally:
        done_q.put("done")
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join()

    assert process.exitcode == 0
