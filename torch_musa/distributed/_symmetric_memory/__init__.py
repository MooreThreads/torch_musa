"""symmetric memory related patches and operators"""

# pylint: disable=unused-argument,C0301
from datetime import timedelta
from typing import Callable, List
import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed.distributed_c10d import _get_group_size_by_name
from torch.distributed._symmetric_memory import (
    rendezvous,
    get_symm_mem_workspace,
    enable_symm_mem_for_group,
    _group_name_to_workspace_tensor,
    _SymmetricMemory,
    _get_backend_stream,
    lib,
    _Work,
)


class Work(_Work):
    def __init__(self) -> None:
        super().__init__()
        self.event = torch.musa.Event()
        self.event.record()

    def wait(self, timeout: timedelta = timedelta(seconds=0)) -> bool:
        self.event.wait()
        return True


_musa_backend_streams: dict[int, torch.musa.Stream] = {}

_musa_multi_backend_streams: dict[int, List[torch.musa.Stream]] = {}


def _musa_get_backend_stream(priority: int = 0) -> torch.musa.Stream:
    if priority not in _musa_backend_streams:
        _musa_backend_streams[priority] = torch.musa.Stream(priority=priority)
    return _musa_backend_streams[priority]


def _musa_get_multi_backend_stream(
    priority: int = 0, index: int = 0
) -> List[torch.musa.Stream]:
    """one priority corresponds to multiple backend streams"""
    if priority not in _musa_multi_backend_streams:
        _musa_multi_backend_streams[priority] = [torch.musa.Stream(priority=priority)]
    else:
        num_stream = len(_musa_multi_backend_streams[priority])
        assert index <= num_stream
        if index == num_stream:
            _musa_multi_backend_streams[priority].append(
                torch.musa.Stream(priority=priority)
            )

    assert index <= len(_musa_multi_backend_streams[priority])
    return _musa_multi_backend_streams[priority][index]


def _musa_memcpy_async(
    dst: torch.Tensor, src: torch.Tensor, use_ce: bool = True
) -> None:
    """Launch an asynchronous MUSA tensor copy, optionally using the copy engine."""
    torch.musa._MUSAC._musa_memcpy_async(dst, src, use_ce)


def _musa_pipelined_multi_all_gather_and_consume(
    shard: list[torch.Tensor],
    shard_consumer: Callable[[list[torch.Tensor], int], None],
    ag_out: list[torch.Tensor],
    group_name: str,
    ag_out_needed: bool = True,
) -> None:
    """
    Pipeline symmetric-memory all-gather copies and consume each gathered shard.

    This has the same logical contract as PyTorch's `_pipelined_multi_all_gather_and_consume`:
    publish every local shard to the symmetric-memory workspace, copy each remote rank's shard into the
    corresponding `ag_out` chunk, and call `shard_consumer` in rank-step order.

    Different from PyTorch's implementation, all `shard_consumer` calls run on the
    compute stream instead of alternating copy-and-consume work between the
    compute stream and a single backend stream.  Local publication and the
    publish barrier are serialized on the compute stream, then remote CE copies
    are enqueued as early as possible on high-priority backend streams (one
    stream for small groups, three streams when `group_size >= 4`).  The compute
    stream waits for each remote copy's event immediately before consuming that
    shard, preserving the original consumption order.

    For rank 0 with `world_size == 4`, the pipeline is:

        time ---->
        compute : [publish][barrier][ready][consume r0][wait e1][consume r1][wait e2][consume r2][wait e3][consume r3]

        backend0:                           [wait ready][CE copy r1][event1]

        backend1:                           [wait ready][CE copy r2......][event2]

        backend2:                           [wait ready][CE copy r3..............][event3]

        overlap :                           copy r1/r2/r3 run while compute
                                            consumes r0, r1, and r2.

    `p2p_ready` is recorded before consuming rank 0, so the three remote copies
    can overlap with the local or previous-rank `shard_consumer` work while the
    compute stream still consumes rank 0, 1, 2, and 3 serially.

    This scheduling is faster for the fused all-gather + matmul path because it
    avoids overlapping multiple `shard_consumer` matmuls with each other, which
    otherwise makes the matmuls contend for compute resources and can lower or
    destabilize their throughput.  At the same time, CE-based remote copies are
    still overlapped with local or previous-shard matmul, and multiple backend
    streams give independent remote copies a chance to use more copy/link
    parallelism, reducing the copy bottleneck.
    """
    p2p_workspace_size_req = 0
    for x in shard:
        p2p_workspace_size_req += x.numel() * x.element_size()
    symm_mem = musa_get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank
    # torch.musa.synchronize()  # sync here even gives betters perf in e2e training ?
    # TODO(mingyuan.wang): i think this barrier is not necessary
    symm_mem.barrier(channel=0)

    compute_stream = torch.musa.current_stream()
    remote_copy_stream_count = 1
    if group_size >= 4:
        remote_copy_stream_count = 3

    backend_streams = [_musa_get_multi_backend_stream(priority=-1, index=0)]
    for stream_idx in range(1, remote_copy_stream_count):
        backend_streams.append(
            _musa_get_multi_backend_stream(priority=-1, index=stream_idx)
        )
    backend_stream = backend_streams[0]

    for x, y in zip(shard, ag_out):
        assert x.is_contiguous(), (
            "_pipelined_all_gather_and_consume: all tensors "
            "in `shard` must be contiguous"
        )
        assert y.is_contiguous(), (
            "_pipelined_all_gather_and_consume: all tensors "
            "in `ag_out` must be contiguous"
        )
        assert x.shape[0] * group_size == y.shape[0]
        assert x.shape[1:] == y.shape[1:]

    def copy_shard(dst: list[torch.Tensor], src: list[torch.Tensor]) -> None:
        """Copy matching shard tensors through asynchronous CE copies."""
        for d, s in zip(dst, src):
            _musa_memcpy_async(d, s, use_ce=True)

    def get_p2p_bufs(remote_rank: int) -> list[torch.Tensor]:
        """Return symmetric-memory P2P buffers for one rank's shard list."""
        offset_bytes = 0
        bufs = []
        for x in shard:
            buf = symm_mem.get_buffer(
                remote_rank,
                x.shape,
                x.dtype,
                storage_offset=offset_bytes // x.element_size(),
            )
            bufs.append(buf)
            offset_bytes += buf.numel() * buf.element_size()
        return bufs

    local_p2p_bufs = get_p2p_bufs(rank)

    shards: list[list[torch.Tensor]] = [[] for _ in range(group_size)]
    for x in ag_out:
        for i, y in enumerate(x.chunk(group_size)):
            shards[i].append(y)

    with compute_stream:
        # Keep local publication and barrier serialized with local matmul.
        for d, s in zip(local_p2p_bufs, shard):
            d.copy_(s)
        # wait all other ranks prepare local buffer done
        symm_mem.barrier(channel=1)

        # the remote copy will be launched on other streams
        p2p_ready = torch.musa.Event()
        p2p_ready.record()
        shard_consumer(shard, rank)

    copy_done_events: list[torch.musa.Event | None] = [None for _ in range(group_size)]
    copy_stream_ready = [False for _ in backend_streams]

    def enqueue_remote_copy(step: int) -> None:
        """Queue one remote shard copy and record its completion event."""
        remote_rank = (step + rank) % group_size
        remote_p2p_bufs = get_p2p_bufs(remote_rank)
        stream_idx = (step - 1) % len(backend_streams)
        copy_stream = backend_streams[stream_idx]
        if not copy_stream_ready[stream_idx]:
            copy_stream.wait_event(p2p_ready)
            copy_stream_ready[stream_idx] = True

        event = torch.musa.Event()
        with copy_stream:
            copy_shard(dst=shards[remote_rank], src=remote_p2p_bufs)
            event.record()
        copy_done_events[step] = event

    # launch remote copy ASAP
    for step in range(1, group_size):
        enqueue_remote_copy(step)

    for step in range(1, group_size):
        event = copy_done_events[step]
        assert event is not None
        compute_stream.wait_event(event)
        remote_rank = (step + rank) % group_size
        shard_consumer(shards[remote_rank], remote_rank)

    if ag_out_needed:
        # Opportunistically overlap it with the last shard_consumer.
        with backend_stream:
            copy_shard(dst=shards[rank], src=shard)

    for stream in backend_streams:
        compute_stream.wait_stream(stream)
    symm_mem.barrier(channel=0)


def musa_get_symm_mem_workspace(group_name: str, min_size: int) -> _SymmetricMemory:
    """
    Get the symmetric memory workspace associated with the process group. If
    ``min_size`` is greater than the workspace associated with ``group_name``,
    the workspace will be re-allocated and re-rendezvous'd.

    Args:
        group_name (str): the name of the process group.
        min_size (int): the size requirement for the workspace in bytes.

    Returns:
        _SymmetricMemory: the symmetric memory workspace associated with the
        group.
    """
    enable_symm_mem_for_group(group_name)

    tensor = _group_name_to_workspace_tensor.get(group_name)
    size = tensor.numel() * tensor.element_size() if tensor is not None else 0
    if tensor is None or size < min_size:
        if torch.musa.is_current_stream_capturing():
            curr_size = 0 if tensor is None else tensor.numel() * tensor.element_size()
            raise RuntimeError(
                f"get_symm_mem_workspace(): the requested size ({min_size} bytes) "
                "is greater than the size of the currently allocated workspace "
                f"({curr_size} bytes). It's currently not possible to expand the "
                "workspace size during graph capture. Please invoke "
                f'`get_symm_mem_workspace(group_name="{group_name}", '
                f'min_size="{min_size}")` before initiating the graph capture '
                "and try again."
            )
        tensor = _SymmetricMemory.empty_strided_p2p(
            (max(size, min_size),),
            [1],
            torch.uint8,
            torch.device(f"musa:{torch.musa.current_device()}"),
            group_name,
        )
        _group_name_to_workspace_tensor[group_name] = tensor
    return _SymmetricMemory.rendezvous(tensor)


lib.define(
    "_low_contention_allreduce(Tensor(a!) tensor, str reduce_op, "
    "str group_name) -> Tensor(a!)"
)


@torch.library.impl(lib, "_low_contention_allreduce", "Meta")
def _low_contention_allreduce_meta(
    tensor: torch.Tensor,
    reduce_op: str,
    group_name: str,
) -> torch.Tensor:
    return tensor


@torch.library.impl(lib, "_low_contention_allreduce", "PrivateUse1")
def _low_contention_allreduce(
    tensor: torch.Tensor,
    reduce_op: str,
    group_name: str,
) -> torch.Tensor:
    """
    Performs an in-place all-reduce with symmetric memory in a low-contention
    fashion.

    When ``tensor`` is already in symmetric memory, the collective operates on
    it directly. Otherwise, the input is copied into the symmetric-memory
    workspace before the collective and the reduced result is copied back.
    """
    symm_mem = rendezvous(tensor, group_name)
    input_is_symm_mem = symm_mem is not None
    if symm_mem is None:
        symm_mem = musa_get_symm_mem_workspace(
            group_name, tensor.numel() * tensor.element_size()
        )

    input_buf = tensor
    if not input_is_symm_mem:
        input_buf = symm_mem.get_buffer(
            symm_mem.rank,
            tensor.shape,
            tensor.dtype,
        )

    backend_stream = _musa_get_backend_stream()
    backend_stream.wait_stream(torch.musa.current_stream())
    with backend_stream:
        if not input_is_symm_mem:
            input_buf.copy_(tensor)
        torch.ops.symm_mem.low_contention_allreduce(
            input_buf,
            reduce_op,
            group_name,
        )
        if not input_is_symm_mem:
            tensor.copy_(input_buf)
        torch._C._distributed_c10d._register_work(tensor, Work())
        return tensor


@torch.library.impl(lib, "_low_contention_all_gather", "PrivateUse1")
def _low_contention_all_gather(
    tensor: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    """
    Performs all-gather with symmetric memory in a low-contention fashion.

    When `tensor` is already in symmetric memory:
        - The collective is carried out without using SMs.
        - No symmetric memory workspace is required.

    When `tensor` is not in symmetric memory:
        - An extra SM-based copy is performed to copy the input data into the
          symmetric memory workspace.
        - Symmetric memory workspace size requirement: the size of `tensor`.
    """
    symm_mem = rendezvous(tensor, group_name)
    if symm_mem is not None:
        input_is_symm_mem = True
    else:
        symm_mem = musa_get_symm_mem_workspace(
            group_name, tensor.numel() * tensor.element_size()
        )
        input_is_symm_mem = False

    rank = symm_mem.rank
    world_size = symm_mem.world_size

    output = tensor.new_empty(tensor.shape[0] * world_size, *tensor.shape[1:])

    backend_stream = _musa_get_backend_stream()

    backend_stream.wait_stream(torch.musa.current_stream())
    with backend_stream:
        if not input_is_symm_mem:
            local_buf = symm_mem.get_buffer(rank, tensor.shape, tensor.dtype)
            # TODO: this copy still uses MP resources, change to leverage copy engine instead
            local_buf.copy_(tensor)

        torch.ops.symm_mem.low_contention_all_gather(
            output, tensor if input_is_symm_mem else local_buf, group_name
        )
        torch._C._distributed_c10d._register_work(output, Work())
        return output


def _musa_pipelined_produce_and_all2all(
    chunk_producer: Callable[[int, torch.Tensor], None],
    output: torch.Tensor,
    group_name: str,
) -> None:
    """
    Pipeline symmetric-memory all-to-all production and remote copies.

    This has the same logical contract as PyTorch's `_pipelined_produce_and_all2all`:
    produce one output chunk for each destination rank, publish every remote chunk
    through the symmetric-memory workspace, copy each peer's published chunk into
    the corresponding `output` chunk, and write the local rank's chunk directly.

    Different from the original copy-and-consume style pipeline, all
    `chunk_producer` calls run on the compute stream in step order.  Each remote
    chunk is produced into a dedicated P2P workspace slot, published with a
    per-slot barrier, then copied back from the peer's matching slot by one of
    copy streams.  The compute stream only waits for the copy streams at the end,
    so remote CE copies can overlap with later chunk production and remote CE copies
    at different stream can overlap with each other.

    For rank 0 with `world_size == 4`, the pipeline is:

        time ---->
        compute: [produce r1][b0][ready0][produce r2][b1][ready1][produce r3][b2][ready2][produce r0][wait copies][b0]

        copy0  :                    [wait ready0][CE copy from r3 slot0]

        copy1  :                                               [wait ready1][CE copy from r2 slot1]

        copy2  :                                                                         [wait ready2][CE copy from r1 slot1]

        overlap:                    copy from r3 runs while compute produces r2/r3,
                                    and copy from r2/r1 runs while compute produces later chunks.

    The per-slot barrier channels make a peer's workspace slot visible before the
    corresponding copy stream reads it.  The produced event is a local stream
    dependency from compute to copy; the symmetric-memory barrier is the cross-rank
    publish point for the slot.
    """
    group_size = _get_group_size_by_name(group_name)
    remote_copy_stream_count = group_size - 1
    if group_size >= 4:
        remote_copy_stream_count = 3
    out_chunks = output.chunk(group_size)
    # TODO: verify memory usage for `group_size - 1` p2p_slots
    num_p2p_slots = max(group_size - 1, 1)
    p2p_workspace_size_req = (
        out_chunks[0].numel() * out_chunks[0].element_size() * num_p2p_slots
    )

    symm_mem = musa_get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    rank = symm_mem.rank
    compute_stream = torch.musa.current_stream()
    copy_streams = [
        _musa_get_multi_backend_stream(priority=-1, index=stream_idx)
        for stream_idx in range(remote_copy_stream_count)
    ]
    symm_mem.barrier(channel=0)

    def get_p2p_buf(rank: int, idx: int) -> torch.Tensor:
        assert 0 <= idx < num_p2p_slots
        offset = out_chunks[0].numel() * idx
        return symm_mem.get_buffer(
            rank, out_chunks[0].shape, out_chunks[0].dtype, offset
        )

    local_p2p_bufs = [get_p2p_buf(rank, idx) for idx in range(num_p2p_slots)]

    for step in range(1, group_size):
        dst_rank = (rank + step) % group_size
        remote_rank = (rank - step) % group_size
        p2p_buf_idx = step - 1
        p2p_buf = local_p2p_bufs[p2p_buf_idx]
        remote_p2p_buf = get_p2p_buf(remote_rank, p2p_buf_idx)
        copy_stream = copy_streams[p2p_buf_idx % remote_copy_stream_count]

        ready_channel = p2p_buf_idx

        with compute_stream:
            chunk_producer(dst_rank, p2p_buf)
            symm_mem.barrier(channel=ready_channel)
            produced_event = torch.musa.Event()
            produced_event.record()

        copy_stream.wait_event(produced_event)
        with copy_stream:
            _musa_memcpy_async(out_chunks[remote_rank], remote_p2p_buf, use_ce=True)

    chunk_producer(rank, out_chunks[rank])
    for stream in copy_streams:
        compute_stream.wait_stream(stream)
    symm_mem.barrier(channel=0)


@torch.library.impl(lib, "_low_contention_reduce_scatter", "PrivateUse1")
def _low_contention_reduce_scatter(
    tensor: torch.Tensor,
    reduce_op: str,
    group_name: str,
):
    """
    Performs reduce-scatter with symmetric memory in a low-contention fashion.


    """
    symm_mem = rendezvous(tensor, group_name)
    backend_stream = _musa_get_backend_stream()

    if symm_mem is not None:
        rank = symm_mem.rank
        world_size = symm_mem.world_size
        assert tensor.shape[0] % world_size == 0
        output = torch.empty(
            (tensor.shape[0] // world_size, *tensor.shape[1:]),
            device=tensor.device,
            dtype=tensor.dtype,
        )

        backend_stream.wait_stream(torch.musa.current_stream())
        with backend_stream:
            torch.ops.symm_mem.low_contention_reduce_scatter(
                output,
                tensor,
                reduce_op,
                group_name,
            )
            torch._C._distributed_c10d._register_work(output, Work())
            return output
    else:
        world_size = _get_group_size_by_name(group_name)
        assert tensor.shape[0] % world_size == 0
        workspace = musa_get_symm_mem_workspace(
            group_name, tensor.numel() * tensor.element_size() // world_size
        )
        rank = workspace.rank
        output_shape = (tensor.shape[0] // world_size, *tensor.shape[1:])
        output_buf = workspace.get_buffer(rank, output_shape, tensor.dtype)

        backend_stream = _musa_get_backend_stream()
        backend_stream.wait_stream(torch.musa.current_stream())
        with backend_stream:
            torch.ops.symm_mem.low_contention_reduce_scatter(
                output_buf,
                tensor,
                reduce_op,
                group_name,
            )
            torch._C._distributed_c10d._register_work(output_buf, Work())

            return output_buf


def _apply_symmetric_memory_patch() -> None:
    """
    Apply the _symmetric_memory patch
    """
    torch.distributed._symmetric_memory._low_contention_all_gather = (
        _low_contention_all_gather
    )
    torch.distributed._symmetric_memory._low_contention_reduce_scatter = (
        _low_contention_reduce_scatter
    )
    torch.distributed._symmetric_memory._low_contention_allreduce = (
        _low_contention_allreduce
    )
    torch.distributed._symmetric_memory._pipelined_multi_all_gather_and_consume = (
        _musa_pipelined_multi_all_gather_and_consume
    )
    torch.distributed._symmetric_memory._pipelined_produce_and_all2all = (
        _musa_pipelined_produce_and_all2all
    )
    torch.distributed._symmetric_memory.get_symm_mem_workspace = (
        musa_get_symm_mem_workspace
    )
