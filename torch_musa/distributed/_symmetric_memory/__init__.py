"""symmetric memory related patches and operators"""

# pylint: disable=unused-argument
from datetime import timedelta
import torch
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


def _musa_get_backend_stream(priority: int = 0) -> torch.musa.Stream:
    if priority not in _musa_backend_streams:
        _musa_backend_streams[priority] = torch.musa.Stream(priority=priority)
    return _musa_backend_streams[priority]


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
    torch.distributed._symmetric_memory.get_symm_mem_workspace = (
        musa_get_symm_mem_workspace
    )
