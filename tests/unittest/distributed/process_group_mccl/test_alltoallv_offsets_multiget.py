"""ProcessGroupMCCL AllToAllV offsets MultiGet integration tests."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_musa  # pylint: disable=wrong-import-position
from torch_musa import testing  # pylint: disable=wrong-import-position


_WORLD_SIZE = 2
_SPLIT_MATRICES = (
    ((1, 2), (3, 4)),
    ((0, 3), (2, 0)),
)


def _run_alltoallv_case(rank, split_matrix):
    input_split_sizes = list(split_matrix[rank])
    output_split_sizes = [splits[rank] for splits in split_matrix]
    input_numel = sum(input_split_sizes)

    input_tensor = torch.arange(
        rank * 100,
        rank * 100 + input_numel,
        dtype=torch.int64,
        device=f"musa:{rank}",
    )
    output_tensor = torch.empty(
        sum(output_split_sizes), dtype=input_tensor.dtype, device=input_tensor.device
    )

    dist.all_to_all_single(
        output_tensor,
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )

    expected_values = []
    for src_rank, src_splits in enumerate(split_matrix):
        offset = sum(src_splits[:rank])
        expected_values.extend(
            range(
                src_rank * 100 + offset,
                src_rank * 100 + offset + src_splits[rank],
            )
        )
    expected = torch.tensor(
        expected_values, dtype=output_tensor.dtype, device=output_tensor.device
    )
    torch.testing.assert_close(output_tensor, expected)


def _alltoallv_worker(rank, world_size, init_method):
    os.environ["TORCH_MCCL_BLOCKING_WAIT"] = "1"

    try:
        torch_musa.set_device(rank)
        dist.init_process_group(
            "mccl",
            rank=rank,
            world_size=world_size,
            init_method=init_method,
            timeout=timedelta(seconds=60),
        )
        for split_matrix in _SPLIT_MATRICES:
            _run_alltoallv_case(rank, split_matrix)
        torch.musa.synchronize(rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@testing.skip_if_not_multiple_musa_device
@pytest.mark.parametrize("store_kind", ("tcp", "file"))
def test_alltoallv_offsets_multiget(store_kind, tmp_path):
    """Validate offsets with extended and fallback Store APIs."""
    if not dist.is_mccl_available():
        pytest.skip("MCCL backend is not available")
    if torch.musa.mccl.version() < (2, 28, 9):
        pytest.skip("MCCL AllToAllV requires MCCL >= 2.28.9")
    if torch.musa.device_count() < _WORLD_SIZE:
        pytest.skip("Need at least 2 MUSA devices")

    if store_kind == "tcp":
        master_addr, master_port = testing.gen_ip_port()
        init_method = f"tcp://{master_addr}:{master_port}"
    else:
        init_method = f"file://{tmp_path / 'alltoallv_offsets'}"

    mp.spawn(
        _alltoallv_worker,
        args=(
            _WORLD_SIZE,
            init_method,
        ),
        nprocs=_WORLD_SIZE,
        join=True,
    )
