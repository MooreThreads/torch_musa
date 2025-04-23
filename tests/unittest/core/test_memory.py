"""Unittest for memory APIs."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, unused-variable, unexpected-keyword-arg
# pylint: disable=invalid-name, import-outside-toplevel, unspecified-encoding, too-many-nested-block, unused-argument
# pylint: disable=too-many-nested-blocks, unrecognized-inline-option

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torch_musa
from torch_musa import testing
from torch_musa.testing import get_cycles_per_ms


def test_single_op():
    """Test a single operator"""
    torch.musa.empty_cache()
    torch.musa.reset_peak_stats()
    input_data = torch.randn(1024, 1024).to("musa")
    torch.matmul(input_data, input_data)

    m_allocated = torch.musa.max_memory_allocated()
    m_reserved = torch.musa.max_memory_reserved()
    allocated = torch.musa.memory_allocated()
    reserved = torch.musa.memory_reserved()
    assert m_allocated == 8 * 1024 * 1024  # maximally allocated 8MB
    assert m_reserved == 20 * 1024 * 1024  # maximally reserved 20MB
    assert allocated == 4 * 1024 * 1024  # allocated 4MB
    assert reserved == 20 * 1024 * 1024  # reserved 20MB


def test_multiple_ops():
    """Test multiple ops"""
    torch.musa.empty_cache()
    torch.musa.reset_peak_stats()
    input_data = torch.randn(1024 * 4, 1024 * 4).to(
        "musa"
    )  # 16 * 1024 * 1024 * 4 = 64MB
    tensor_a = torch.matmul(input_data, input_data)
    torch.matmul(tensor_a, tensor_a)
    summary = torch.musa.memory_summary()
    assert "reserved" in summary

    stats = torch.musa.memory_stats()
    assert isinstance(stats, dict)

    m_allocated = torch.musa.max_memory_allocated()
    m_reserved = torch.musa.max_memory_reserved()
    allocated = torch.musa.memory_allocated()
    reserved = torch.musa.memory_reserved()
    assert m_allocated == 192 * 1024 * 1024  # maximally allocated 192MB
    assert m_reserved == 192 * 1024 * 1024  # maximally reserved 192MB
    assert allocated == 128 * 1024 * 1024  # allocated 128MB
    assert reserved == 192 * 1024 * 1024  # reserved 192MB


def test_single_device_ops():
    """Test memory stats ops when specify single device"""
    torch.musa.empty_cache()
    torch.musa.reset_peak_stats()
    input_data_small = torch.randn(1024 * 4, 1024 * 4).to(
        "musa:0"
    )  # 16 * 1024 * 1024 * 4 = 64MB
    input_data_large = torch.randn(1024 * 8, 1024 * 8).to(
        "musa:0"
    )  # 64 * 1024 * 1024 * 4 = 256MB
    del input_data_large  # free 256MB on device 0
    summary = torch.musa.memory_summary(0)
    assert "reserved" in summary

    stats = torch.musa.memory_stats(0)
    assert isinstance(stats, dict)

    m_allocated = torch.musa.max_memory_allocated(0)
    m_reserved = torch.musa.max_memory_reserved(0)
    allocated = torch.musa.memory_allocated(0)
    reserved = torch.musa.memory_reserved(0)
    assert m_allocated == 320 * 1024 * 1024  # maximally allocated 320MB
    assert m_reserved == 320 * 1024 * 1024  # maximally reserved 320MB
    assert allocated == 64 * 1024 * 1024  # allocated 64MB
    assert reserved == 320 * 1024 * 1024  # reserved 320MB


@testing.skip_if_not_multiple_musa_device
def test_all_devices_ops():
    """Test memory stats ops when specify all devices"""
    torch.musa.empty_cache()
    torch.musa.reset_peak_stats()
    tensor_0 = torch.randn(1024 * 4, 1024 * 4).to(
        "musa:0"
    )  # 16 * 1024 * 1024 * 4 = 64MB
    tensor_1 = torch.randn(1024 * 8, 1024 * 8).to(
        "musa:1"
    )  # 64 * 1024 * 1024 * 4 = 256MB
    summary = torch.musa.memory_summary(all_device=True)
    assert "reserved" in summary
    assert "[ALL]" in summary

    stats = torch.musa.memory_stats_all()
    assert isinstance(stats, dict)

    m_allocated = torch.musa.max_memory_allocated(all_device=True)
    m_reserved = torch.musa.max_memory_reserved(all_device=True)
    allocated = torch.musa.memory_allocated(all_device=True)
    reserved = torch.musa.memory_reserved(all_device=True)
    assert m_allocated == 256 * 1024 * 1024  # maximally allocated 256MB
    assert m_reserved == 256 * 1024 * 1024  # maximally reserved 256MB
    assert allocated == (64 + 256) * 1024 * 1024  # allocated 320MB
    assert reserved == (64 + 256) * 1024 * 1024  # reserved 320MB


def test_caching_pinned_memory():
    """Test caching host allocator functions."""
    # check that allocations are re-used after deletion
    pinned_tensor = torch.tensor([1]).pin_memory("musa")
    ptr = pinned_tensor.data_ptr()
    del pinned_tensor
    pinned_tensor = torch.tensor([1]).pin_memory("musa")
    assert pinned_tensor.data_ptr() == ptr, "allocation not reused."

    # check that the allocation is not re-used if it's in-use by a copy
    gpu_tensor = torch.tensor([0], device="musa")
    torch.musa._sleep(int(1000 * get_cycles_per_ms()))  # delay the copy by 1s
    gpu_tensor.copy_(pinned_tensor, non_blocking=True)
    del pinned_tensor
    pinned_tensor = torch.tensor([1]).pin_memory("musa")
    assert pinned_tensor.data_ptr() != ptr, "allocation re-used too soon."
    assert list(gpu_tensor) == [1]

    # check pin memory copy with different dtypes
    gpu_tensor = torch.tensor([0.0], device="musa", dtype=torch.float32)
    pinned_tensor = torch.tensor([1], dtype=torch.int8).pin_memory("musa")
    gpu_tensor.copy_(pinned_tensor, non_blocking=True)
    assert list(gpu_tensor) == [1]

    # check pin memory D2H copy
    gpu_tensor = torch.tensor([1], device="musa")
    pinned_tensor = torch.tensor([0], dtype=torch.int32).pin_memory("musa")
    pinned_tensor.copy_(gpu_tensor)
    assert list(pinned_tensor) == [1]


class DictDataset(Dataset):
    """Dictionary data loader."""

    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return {
            "a_tensor": torch.empty(4, 2).fill_(ndx),
            "another_dict": {
                "a_number": ndx,
            },
        }


def test_pin_memory_dataloader():
    """Test dataloader with pin memory."""
    dataset = DictDataset()
    loader = DataLoader(
        dataset, batch_size=2, pin_memory=True, pin_memory_device="musa"
    )
    for sample in loader:
        assert sample["a_tensor"].is_pinned("musa")
        assert sample["another_dict"]["a_number"].is_pinned("musa")


@testing.skip_if_not_multiple_musa_device
def test_pin_memory_dataloader_non_zero_device():
    """Test dataloader with pin memory on non-zero gpu."""
    dataset = DictDataset()
    loader = DataLoader(
        dataset, batch_size=2, pin_memory=True, pin_memory_device="musa:1"
    )
    for sample in loader:
        assert sample["a_tensor"].is_pinned("musa:1")
        assert sample["another_dict"]["a_number"].is_pinned("musa:1")


def test_pin_memory():
    """Test pin memory"""
    x = torch.randn(20, 20)
    assert not x.is_pinned()
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")

    pinned = x.pin_memory()
    assert pinned.is_pinned()
    # The original tensor should be different from the pinned one.
    assert pinned.data_ptr() != x.data_ptr()
    # Pin already pinned tensor has no side-effect
    assert pinned is pinned.pin_memory()
    assert pinned.data_ptr() == pinned.pin_memory().data_ptr()


def test_pin_memory_empty():
    """Test pin memory for empty interface"""
    x1 = torch.empty((3, 5), pin_memory=True)
    x2 = x1.to("musa", non_blocking=True)
    assert x1.is_pinned()
    assert x2.is_musa

    # TODO(mt-ai): enable the following test when empty_strided is supported
    # x1 = torch.empty_strided((2, 3), (1, 2), pin_memory=True)
    # x2 = x1.to("musa", non_blocking=True)
    # assert x1.is_pinned()
    # assert x2.is_musa


def test_pin_memory_with_operator():
    """Test pin memory using operator constructors"""

    def _generate_tensors(**kwargs):
        return [
            torch.arange(2, 3, **kwargs),
            torch.empty(2, **kwargs),
            torch.eye(2, **kwargs),
            torch.ones(2, **kwargs),
            torch.rand(3, **kwargs),
            torch.randn(2, 3, **kwargs),
            torch.randperm(3, **kwargs),
            torch.tensor([2, 3], **kwargs),
            torch.zeros(3, **kwargs),
        ]

    pinned_tensors = _generate_tensors(pin_memory=True)
    for x in pinned_tensors:
        assert x.is_pinned()

        x1 = x.to("musa")
        assert x1.is_musa

    tensors = _generate_tensors()
    for x in tensors:
        assert not x.is_pinned()


def test_set_per_process_memory_fraction():
    # test invalid fraction value.
    with pytest.raises(TypeError, match="Invalid type"):
        torch.musa.set_per_process_memory_fraction(int(1))
    with pytest.raises(ValueError, match="Invalid fraction value"):
        torch.musa.set_per_process_memory_fraction(-0.1)
    with pytest.raises(ValueError, match="Invalid fraction value"):
        torch.musa.set_per_process_memory_fraction(2.0)

    tensor = torch.zeros(1024, device="musa")
    torch.musa.empty_cache()
    total_memory = torch.musa.get_device_properties(0).total_memory
    torch.musa.set_per_process_memory_fraction(0.5, 0)

    # test 0.499 allocation is ok.
    # TODO(Xiaokang Shang): Waitint for MUSA support contiguous virtual address.
    application = int(total_memory * 0.499) - torch.musa.max_memory_reserved()
    tmp_tensor = torch.empty(application, dtype=torch.int8, device="musa")
    del tmp_tensor
    torch.musa.empty_cache()

    application = int(total_memory * 0.5)
    # it will get OOM when try to allocate more than half memory.
    with pytest.raises(RuntimeError, match="out of memory"):
        torch.empty(application, dtype=torch.int8, device="musa")

    # ensure out of memory error doesn't disturb subsequent kernel
    tensor.fill_(1)
    assert (tensor == 1).all()


def test_matmul_memory_use():
    def get_max_used():
        torch.musa.synchronize()
        val = torch.musa.max_memory_allocated()
        torch.musa.reset_peak_memory_stats()
        return val

    a = torch.rand(1, 32, 32, device="musa")
    b = torch.rand(24, 32, 1, device="musa")

    get_max_used()

    torch.matmul(a, b)

    matmul_mem = get_max_used()

    a = a.expand(24, 32, 32)
    torch.matmul(a, b)

    matmul_expand_mem = get_max_used()

    torch.bmm(a, b)

    bmm_mem = get_max_used()

    assert matmul_expand_mem == matmul_mem
    assert bmm_mem == matmul_mem


@pytest.mark.skipif(True, reason="Hang in CI, to be fixed")
def test_memory_snapshot():
    try:
        from random import randint

        torch.musa.memory.empty_cache()
        torch.musa.memory._record_memory_history(True)
        x = torch.rand(311, 411, device="musa")

        # create a bunch of tensors that all will tile into the
        # same segment to  exercise the history merging code
        # 512B is the minimum block size,
        # so we allocate all the tensors to this size to make sure
        # they tile evenly
        tensors = [torch.rand(128, device="musa") for _ in range(1000)]
        while tensors:
            del tensors[randint(0, len(tensors) - 1)]

        # exercise the history trimming code
        torch.rand(128 * 5, device="musa")

        ss = torch.musa.memory._snapshot()
        found_it = False
        for seg in ss["segments"]:
            for b in seg["blocks"]:
                if "history" in b:
                    for h in b["history"]:
                        if h["real_size"] == 311 * 411 * 4:
                            assert "test_memory" in h["frames"][0]["filename"]
                            found_it = True
        assert found_it

        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            torch.musa.memory._save_segment_usage(f.name)
            with open(f.name, "r") as f2:
                assert "test_memory.py" in f2.read()

        del x
        torch.musa.empty_cache()
        ss = torch.musa.memory._snapshot()
        assert ss["device_traces"][0][-1]["action"] == "segment_free"

    finally:
        torch.musa.memory._record_memory_history(False)


def test_memory_snapshot_with_cpp():
    try:
        torch.musa.memory.empty_cache()
        torch.musa.memory._record_memory_history(True, _enable_expensive_cpp=True)
        x = torch.rand(311, 411, device="musa")

        ss = torch.musa.memory._snapshot()["segments"]
        found_it = False
        for seg in ss:
            for b in seg["blocks"]:
                if "history" in b:
                    for h in b["history"]:
                        if h["real_size"] == 311 * 411 * 4:
                            assert len(h["cpp_frames"]) > 0
                            found_it = True
        assert found_it

    finally:
        torch.musa.memory._record_memory_history(False)


def test_notifies_oom():
    x = False

    def cb(device, alloc, device_alloc, device_free):
        nonlocal x
        x = True

    torch.musa._MUSAC._musa_attach_out_of_memory_observer(cb)
    with pytest.raises(RuntimeError, match="out of memory"):
        torch.empty(1024 * 1024 * 1024 * 1024, device="musa")
    assert x


def test_storage_resize():
    test_tensor = torch.randn(1024 * 1024).to("musa")
    test_tensor.storage().resize_(0)
    test_tensor.storage().resize_(test_tensor.numel())
