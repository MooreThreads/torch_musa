"""Test MemPool"""

import ctypes
import threading

import pytest
import torch

from torch_musa.utils.musa_extension import load_inline
from torch_musa.testing.common_utils import spawn_isolated_test


# pylint: disable=C0115,C0116
def test_mempool_id():
    pool1 = torch.musa.graph_pool_handle()
    pool2 = torch.musa.MemPool().id

    assert pool1[0] == pool2[0]
    assert (pool2[1] - pool1[1]) > 0


def test_mempool_context():
    active_pool = torch.musa.MemPoolContext.active_pool()
    assert active_pool is None, "there should be no active pool"

    pool = torch.musa.MemPool()
    ctx = torch.musa.MemPoolContext(pool)
    active_pool = torch.musa.MemPoolContext.active_pool()
    assert active_pool is not None and (pool == active_pool)

    # leave out ctx, active pool is the previous one, i.e., None
    del ctx
    active_pool = torch.musa.MemPoolContext.active_pool()
    assert active_pool is None


def test_mempool_multithread():
    pool_ids, active_pool_ids = [], []

    def create_mempool_and_make_active():
        pool = torch.musa.MemPool()
        pool_ids.extend([pool.id])

        ctx = torch.musa.MemPoolContext(pool)
        active_pool = torch.musa.MemPoolContext.active_pool()
        active_pool_ids.extend([active_pool.id])
        del ctx

    num_threads = 4
    threads = [
        threading.Thread(target=create_mempool_and_make_active)
        for _ in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # each thread should create a unique mempool, since mempool id creation is atomic
    assert len(set(pool_ids)) == 4

    # each thread should have different active mempool, sice the pointer
    # to the active mempool is thread local
    assert len(set(active_pool_ids)) == 4


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
    assert pool.allocator is None

    dummy_allocator, allocator = _get_dummy_allocator()
    pool = torch.musa.MemPool(allocator.allocator())

    assert id(allocator.allocator()) == id(pool.allocator)
    alloc_lib = ctypes.CDLL(dummy_allocator)
    alloc_called_flag = ctypes.c_int.in_dll(alloc_lib, "alloc_called_flag")
    free_called_flag = ctypes.c_int.in_dll(alloc_lib, "free_called_flag")
    assert alloc_called_flag.value == 0
    assert free_called_flag.value == 0

    with torch.musa.use_mem_pool(pool):
        # will route to custom malloc logic
        _ = torch.randn((1,), device="musa")
        assert alloc_called_flag.value == 1


@spawn_isolated_test
def test_memory_snapshot_with_mempool_id():
    torch.musa.empty_cache()
    pool = torch.musa.MemPool()
    data = []

    with torch.musa.use_mem_pool(pool):
        data.append(torch.empty(1024, device="musa"))

    segments = torch.musa.memory_snapshot(pool.id)
    _assert_segments_belong_to_pool(segments, pool)

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
