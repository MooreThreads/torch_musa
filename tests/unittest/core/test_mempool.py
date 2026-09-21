"""Test MemPool"""

import ctypes
import threading

import pytest
import torch

from torch_musa.utils.musa_extension import load_inline
from torch_musa.testing.common_utils import spawn_isolated_test


# pylint: disable=C0115,C0116,C0103,
def test_mempool_id():
    pool1 = torch.musa.graph_pool_handle()
    pool2 = torch.musa.MemPool().id

    assert pool1[0] == pool2[0]
    assert (pool2[1] - pool1[1]) > 0


def test_mempool_multithread():
    pool_ids = []

    def create_mempool_and_make_active():
        pool = torch.musa.MemPool()
        pool_ids.extend([pool.id])

    num_threads = 4
    threads = [
        threading.Thread(target=create_mempool_and_make_active)
        for _ in range(num_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # each thread should create a unique mempool, since mempool id creation is atomic
    assert len(set(pool_ids)) == 4


def _get_dummy_allocator():
    dummy_allocator_source = """
    #include <sys/types.h>
    #include <musa_runtime_api.h>

    extern "C" {
        int alloc_called_flag = 0;
        int free_called_flag = 0;

        void* dummy_alloc(size_t size, int device, void* stream) {
        void* ptr;
        musaMalloc(&ptr, size);
        alloc_called_flag = 1;
        return ptr;
        }

        void dummy_free(void* ptr, size_t size, int device, void* stream) {
        musaFree(ptr);
        free_called_flag = 1;
        }
    }
    """

    dummy_allocator_libname = "dummy_allocator"
    dummy_allocator = load_inline(
        name=dummy_allocator_libname,
        cpp_sources=dummy_allocator_source,
        is_python_module=False,
        keep_intermediates=False,
        verbose=True,
        with_musa=True,
    )
    allocator = torch.musa.memory.MUSAPluggableAllocator(
        dummy_allocator,
        "dummy_alloc",
        "dummy_free",
    )

    return dummy_allocator, allocator


def _assert_segments_belong_to_pool(segments, pool):
    assert segments, "Expected pool snapshot to contain at least one segment"
    for segment in segments:
        assert segment["segment_pool_id"] == pool.id


@spawn_isolated_test
def test_mempool_with_allocator():
    pool = torch.musa.MemPool()

    dummy_allocator, allocator = _get_dummy_allocator()
    pool = torch.musa.MemPool(allocator.allocator())

    assert pool.use_count() == 1

    alloc_lib = ctypes.CDLL(dummy_allocator)
    alloc_called_flag = ctypes.c_int.in_dll(alloc_lib, "alloc_called_flag")
    free_called_flag = ctypes.c_int.in_dll(alloc_lib, "free_called_flag")
    assert alloc_called_flag.value == 0
    assert free_called_flag.value == 0

    nelem_1mb = 1024 * 1024 // 4

    with torch.musa.use_mem_pool(pool):
        out_0 = torch.randn(nelem_1mb, device="musa")
        assert pool.use_count() == 2

    assert pool.use_count() == 1
    assert alloc_called_flag.value == 1

    out_non_pool = torch.empty(nelem_1mb, device="musa")
    assert out_non_pool is not None

    with torch.musa.use_mem_pool(pool):
        assert len(pool.snapshot()) == 1

        out_1 = torch.randn(nelem_1mb, device="musa")

        assert len(pool.snapshot()) == 1

        out_2 = torch.randn(nelem_1mb, device="musa")

        assert len(pool.snapshot()) == 2

    assert len(pool.snapshot()) == 2

    del out_0, out_1, out_2
    del pool

    assert free_called_flag.value == 1


@spawn_isolated_test
def test_memory_snapshot_with_mempool_id():
    torch.musa.empty_cache()
    pool = torch.musa.MemPool()
    data = []

    with torch.musa.use_mem_pool(pool):
        data.append(torch.empty(1024, device="musa"))

    segments = torch.musa.memory_snapshot(pool.id)
    _assert_segments_belong_to_pool(segments, pool)

    segments_without_traces = torch.musa.memory_snapshot(pool.id, include_traces=False)
    _assert_segments_belong_to_pool(segments_without_traces, pool)

    all_segment_addresses = {
        segment["address"] for segment in torch.musa.memory_snapshot()
    }
    pool_segment_addresses = {segment["address"] for segment in segments}
    assert pool_segment_addresses.issubset(all_segment_addresses)


@spawn_isolated_test
def test_mempool_snapshot():
    torch.musa.empty_cache()
    pool = torch.musa.MemPool()
    data = []

    with torch.musa.use_mem_pool(pool):
        data.append(torch.empty(1024, device="musa"))

    assert pool.snapshot() == torch.musa.memory_snapshot(pool.id)
    assert pool.snapshot(include_traces=False) == torch.musa.memory_snapshot(
        pool.id, include_traces=False
    )
    _assert_segments_belong_to_pool(pool.snapshot(), pool)


@spawn_isolated_test
def test_use_mem_pool_current_thread():
    torch.musa.empty_cache()
    pool = torch.musa.MemPool()
    data = []

    with torch.musa.use_mem_pool(pool):
        data.append(torch.empty(1024, device="musa"))
        snapshot_before = pool.snapshot()

        def allocate_outside_current_thread():
            data.append(torch.empty(2048, device="musa"))
            torch.musa.synchronize()

        thread = threading.Thread(target=allocate_outside_current_thread)
        thread.start()
        thread.join()

        assert pool.snapshot() == snapshot_before


@spawn_isolated_test
def test_use_mem_pool_current_thread_all_streams():
    torch.musa.empty_cache()
    pool = torch.musa.MemPool()
    data = []

    with torch.musa.use_mem_pool(pool):
        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            data.append(torch.empty(1024, device="musa"))
        stream.synchronize()

    _assert_segments_belong_to_pool(pool.snapshot(), pool)


@spawn_isolated_test
def test_use_mem_pool_device_argument():
    torch.musa.empty_cache()
    data = []
    devices = [0, "musa:0", torch.device("musa:0")]

    for device in devices:
        pool = torch.musa.MemPool()
        with torch.musa.use_mem_pool(pool, device=device):
            data.append(torch.empty(16, device="musa"))
        _assert_segments_belong_to_pool(pool.snapshot(), pool)

    pool = torch.musa.MemPool()
    with pytest.raises(ValueError, match="Expected a musa device"):
        with torch.musa.use_mem_pool(pool, device=torch.device("cpu")):
            pass


@spawn_isolated_test
def test_mempool_expandable():
    torch.musa.memory._set_allocator_settings("expandable_segments:True")
    _, allocator = _get_dummy_allocator()
    pool = torch.musa.MemPool(allocator.allocator())

    data = []
    nelem = 1024 * 1024 // 4
    with torch.musa.use_mem_pool(pool):
        data.append(torch.empty(nelem, device="musa"))

    # the second allocation should be in expandable segment
    data.append(torch.empty(nelem, device="musa"))

    segments = torch.musa.memory.memory_snapshot()

    num_expandable_segments = 0
    for segment in segments:
        if segment["is_expandable"]:
            num_expandable_segments += 1

    assert len(segments) == 2, "Expected to have 2 segment"
    assert num_expandable_segments == 1, "Expected to have 1 expandable segment only"


def _setup_mempool_limited_memory_test(additional_allowed_memory_in_mb):
    device = 0
    init_fraction = torch.musa.get_per_process_memory_fraction(device)
    torch.musa.memory.empty_cache()
    mb = 1024 * 1024
    _, all_memory = torch.musa.memory.mem_get_info(device)
    pre_reserved = torch.musa.memory_reserved(device)
    total_allowed = additional_allowed_memory_in_mb * mb + pre_reserved
    fraction_allowed = total_allowed / all_memory
    torch.musa.set_per_process_memory_fraction(fraction_allowed, device)
    return device, init_fraction


def _teardown_mempool_limited_memory_test(device, init_fraction):
    torch.musa.memory.empty_cache()
    torch.musa.set_per_process_memory_fraction(init_fraction, device)


@spawn_isolated_test
def test_tensor_delete_after_allocator_delete():
    dummy_allocator, allocator = _get_dummy_allocator()
    pool = torch.musa.MemPool(allocator.allocator())

    alloc_lib = ctypes.CDLL(dummy_allocator)
    alloc_called_flag = ctypes.c_int.in_dll(alloc_lib, "alloc_called_flag")
    free_called_flag = ctypes.c_int.in_dll(alloc_lib, "free_called_flag")
    assert alloc_called_flag.value == 0
    assert free_called_flag.value == 0

    with torch.musa.use_mem_pool(pool):
        data = torch.empty(4, device="musa")

    assert alloc_called_flag.value == 1
    assert free_called_flag.value == 0

    del pool
    del allocator

    assert free_called_flag.value == 0

    del data
    torch.musa.memory.empty_cache()

    assert free_called_flag.value == 1


@spawn_isolated_test
def test_mempool_limited_memory_with_allocator():
    _, allocator = _get_dummy_allocator()
    pool_do_not_use = torch.musa.MemPool(allocator.allocator())
    pool_use = torch.musa.MemPool(allocator.allocator(), use_on_oom=True)

    nelem_1mb = 1024 * 1024 // 4
    device, init_fraction = _setup_mempool_limited_memory_test(80)

    try:
        with torch.musa.use_mem_pool(pool_do_not_use):
            a = torch.randn(40 * nelem_1mb, device="musa")
        with torch.musa.use_mem_pool(pool_use):
            b = torch.randn(40 * nelem_1mb, device="musa")

        a_dataptr = a.data_ptr()
        b_dataptr = b.data_ptr()

        with pytest.raises(RuntimeError, match="out of memory"):
            torch.randn(40 * nelem_1mb, device="musa")

        del a, b

        c = torch.randn(30 * nelem_1mb, device="musa")
        c_dataptr = c.data_ptr()

        with pytest.raises(RuntimeError, match="out of memory"):
            torch.randn(30 * nelem_1mb, device="musa")

        del c

        assert b_dataptr == c_dataptr

        with torch.musa.use_mem_pool(pool_use):
            e = torch.randn(20 * nelem_1mb, device="musa")
        e_dataptr = e.data_ptr()
        del e

        assert e_dataptr == c_dataptr
        assert a_dataptr != 0
    finally:
        del pool_use, pool_do_not_use, allocator
        _teardown_mempool_limited_memory_test(device, init_fraction)


@spawn_isolated_test
def test_mempool_no_split():
    torch.musa.memory.empty_cache()

    pool_split = torch.musa.MemPool()
    pool_no_split = torch.musa.MemPool(no_split=True)
    nelem_1mb = 1024 * 1024 // 4

    data = []
    with torch.musa.use_mem_pool(pool_split):
        data.append(torch.randn(4 * nelem_1mb, device="musa"))
    with torch.musa.use_mem_pool(pool_no_split):
        data.append(torch.randn(4 * nelem_1mb, device="musa"))

    with torch.musa.use_mem_pool(pool_split):
        data.append(torch.randn(4 * nelem_1mb, device="musa"))
    with torch.musa.use_mem_pool(pool_no_split):
        data.append(torch.randn(4 * nelem_1mb, device="musa"))

    assert len(data) == 4

    if len(pool_no_split.snapshot()) <= len(pool_split.snapshot()):
        raise AssertionError(
            f"Expected no_split pool to have more segments, "
            f"but got {len(pool_no_split.snapshot())} vs {len(pool_split.snapshot())}"
        )

    for seg in pool_no_split.snapshot():
        if len(seg["blocks"]) != 1:
            raise AssertionError(
                f"Expected 1 block in no_split segment, got {len(seg['blocks'])}"
            )

    def count_blocks(pool):
        total = 0
        for seg in pool.snapshot():
            total += len(seg["blocks"])
        return total

    blocks_split = count_blocks(pool_split)
    blocks_no_split = count_blocks(pool_no_split)

    if blocks_no_split >= blocks_split:
        raise AssertionError(
            f"Expected no_split pool to have fewer blocks, "
            f"but got {blocks_no_split} vs {blocks_split}"
        )
