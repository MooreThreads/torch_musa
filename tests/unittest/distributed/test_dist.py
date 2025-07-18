"""Unit test of distributed comm operators"""

# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import, W0611, C0103
import os
import time
import torch
import torch.nn.parallel
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
import pytest
import torch_musa
from torch_musa import testing


def start(rank, world_size):
    ip, port = testing.gen_ip_port()
    if os.getenv("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = ip
    if os.getenv("MASTER_PORT") is None:
        os.environ["MASTER_PORT"] = port
    dist.init_process_group("mccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _test_allgather(dtype=torch.float32):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch_musa.set_device(rank)
    device = torch.device("musa")
    tensor_size = 100
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        tensor_size = 5

    tensor_list = [
        torch.zeros(tensor_size).to(device).to(dtype) for _ in range(world_size)
    ]
    tensor = torch.arange(tensor_size).to(device) + 1 + 2 * rank
    tensor = tensor.to(dtype)

    dist.all_gather(tensor_list, tensor)
    # assert tensors gather to same device.
    assert tensor_list[rank].device == tensor.device
    assert tensor_list[1 - rank].device == tensor.device

    # assert tensor_list has same data with tensor
    for x, y in zip(tensor, tensor_list[rank]):
        assert x == y
    for x, y in zip(tensor, tensor_list[1 - rank]):
        # rank1 == rank0 + 2
        # rank0 + 4*(0.5-0) == rank0 + 2
        # rank1 + 4*(0.5-1) == rank1 - 2
        assert x + 4 * (0.5 - rank) == y


def _test_broacast(dtype=torch.float32):
    rank = dist.get_rank()
    # world_size = dist.get_world_size()
    device = torch.device(f"musa:{rank}")
    tensor_size = 100
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        tensor_size = 5

    if rank == 0:
        tensor = torch.arange(tensor_size).to(device) + 1
    else:
        tensor = torch.zeros(tensor_size).to(device)
    tensor = tensor.to(dtype)

    dist.broadcast(tensor, 0)
    assert tensor.device == device
    for _ in tensor:
        assert _ != 0

    tmp_tensor = torch.tensor([], device=device)
    tmp_tensor = tmp_tensor.to(dtype)
    dist.broadcast(tmp_tensor, src=0, async_op=False)
    for _ in tmp_tensor:
        assert _ != 0


def _test_allreduce(dtype=torch.float32):
    rank = dist.get_rank()
    # world_size = dist.get_world_size()
    tensor_size = 100
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        tensor_size = 10

    device = torch.device(f"musa:{rank}")
    tensor = torch.arange(tensor_size).to(device) + 1 + 2 * rank
    tensor = tensor.to(dtype)
    # rank0 : [1, 2, 3, 4, ...]
    # rank1 : [3, 4, 5, 6, ...]
    dist.all_reduce(tensor, op=ReduceOp.AVG)
    # rank0 : [2, 3, 4, 5, ...]
    # rank1 : [2, 3, 4, 5, ...]

    result = torch.arange(tensor_size).to(device) + 2
    result = result.to(dtype)
    for x, y in zip(tensor, result):
        assert x == y


def _test_reduce(dtype=torch.float32):
    rank = dist.get_rank()
    device = torch.device(f"musa:{rank}")
    tensor_size = 100
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        tensor_size = 6

    tensor = torch.arange(tensor_size).to(device) + 1 + 2 * rank
    tensor = tensor.to(dtype)
    # rank0 : [1, 2, 3, 4, ...]
    # rank1 : [3, 4, 5, 6, ...]
    dist.reduce(tensor, dst=0, op=ReduceOp.AVG)
    # rank0 : reduced to [2, 3, 4, 5, ...]
    # rank1 : No change as [3, 4, 5, 6, ...]
    result = torch.arange(tensor_size, dtype=torch.float32).to(device) + 2 + rank
    result = result.to(dtype)
    for x, y in zip(tensor, result):
        assert x == y


def _test_gather(dtype=torch.float32):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"musa:{rank}")

    tensor_size = 100
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        tensor_size = 5

    tensor_list = [
        torch.zeros(tensor_size).to(device).to(dtype) for _ in range(world_size)
    ]
    tensor = torch.zeros(tensor_size).to(device) + 1 + 2 * rank
    tensor = tensor.to(dtype)

    # Rank0: input is [1...]
    # Rank1: input is [3...]
    if rank == 1:
        dist.gather(tensor, tensor_list, dst=1)
    else:
        dist.gather(tensor, dst=1)
    # Rank1: result is [[1...], [3...]]
    # Rank0: result is [[0...], [0...]]
    result_list = [
        torch.zeros(tensor_size).to(device).to(dtype) + rank * (1 + 2 * _)
        for _ in range(world_size)
    ]

    for T1, T2 in zip(tensor_list, result_list):
        for x, y in zip(T1, T2):
            assert x == y


def _test_scatter(dtype=torch.float32):
    rank = dist.get_rank()
    device = torch.device(f"musa:{rank}")

    tensor_size = 10
    # if dtype == torch.float8_e5m2 or dtype == torch.float8_e4m3fn:
    #    tensor_size = 5

    t_ones = torch.ones(tensor_size).to(device).to(dtype)
    t_fives = torch.ones(tensor_size).to(device).to(dtype) * 5
    output_tensor = torch.zeros(tensor_size).to(device).to(dtype)
    if dist.get_rank() == 0:
        # Assumes world_size of 2.
        # # Only tensors, all of which must be the same size.
        scatter_list = [t_ones, t_fives]
    else:
        scatter_list = None

    dist.scatter(output_tensor, scatter_list, src=0)
    # Rank 0: [1...]
    # Rank 1: [5...]
    result = torch.ones(tensor_size).to(device).to(dtype) + 4 * rank
    for x, y in zip(output_tensor, result):
        assert x == y


def _test_reducescatter(dtype=torch.float32):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch_musa.set_device(rank)
    device = torch.device("musa")
    tensor_size = 10
    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        tensor_size = 6
    # In rankN TensorList[M] is arange() + M + N*Cards
    # Rank0: [1, 2, 3...][2, 3, 4...]
    # Rank1: [3, 4, 5...][4, 5, 6...]
    input_tensor_list = [
        torch.arange(tensor_size).to(device).to(dtype) + 1 + world_size * rank + _
        for _ in range(world_size)
    ]
    output = torch.zeros(tensor_size).to(device).to(dtype)
    dist.reduce_scatter(output, input_tensor_list)
    # Rank0 result: [4, 6, 8 ...]
    # Rank1 result: [6, 8, 10 ...]
    result = (
        torch.arange(tensor_size).to(device).to(dtype) + 2
    ) * world_size + rank * world_size
    for x, y in zip(output, result):
        assert x == y


def _test_all_to_all(dtype=torch.float32):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch_musa.set_device(rank)
    device = torch.device("musa")
    tensor_size = 10

    if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        tensor_size = 5
    # In rankN TensorList[M] is arange() + M + N*Cards
    # Rank0: [1, 2, 3...][2, 3, 4...]
    # Rank1: [3, 4, 5...][4, 5, 6...]
    input_tensor_list = [
        torch.arange(tensor_size).to(device).to(dtype) + 1 + world_size * rank + _
        for _ in range(world_size)
    ]
    output_list = [
        torch.zeros(tensor_size).to(device).to(dtype) for _ in range(world_size)
    ]
    dist.all_to_all(output_list, input_tensor_list)
    # Rank0 result: [1, 2, 3...][3, 4, 5...]
    # Rank1 result: [2, 3, 4...][4, 5, 6...]
    result = [
        torch.arange(tensor_size).to(device).to(dtype) + _ * 2 + 1 + rank
        for _ in range(world_size)
    ]
    for T1, T2 in zip(output_list, result):
        for x, y in zip(T1, T2):
            assert x == y


def _test_barrier():
    rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # device = torch.device(f'musa:{rank}')
    dist.barrier()
    t0 = time.time()
    if rank == 0:
        time.sleep(1)
    dist.barrier()
    t1 = time.time()
    dist.barrier()
    if rank == 1:
        time.sleep(1)
    t2 = time.time()
    assert t1 - t0 > 0.5
    assert t2 - t0 > 0.5 + rank


def _test_broadcast_object():
    rank = dist.get_rank()
    obj = [1, 2, 3, rank]  # a demo obj of (1,2,3,0)
    broadcast_list = [obj if rank == 0 else None]
    dist.broadcast_object_list(broadcast_list, 0)
    if rank != 0:
        obj = broadcast_list[0]
    assert obj == [1, 2, 3, 0]


def _test_demo(demo_fn, world_size, dtype):
    mp.spawn(
        demo_fn,
        args=(
            world_size,
            dtype,
        ),
        nprocs=world_size,
        join=True,
    )


def _test_function(rank, world_size, dtype):
    start(rank, world_size)

    _test_allgather(dtype)
    _test_broacast(dtype)
    _test_allreduce(dtype)
    _test_reducescatter(dtype)
    _test_reduce(dtype)
    _test_gather(dtype)
    _test_scatter(dtype)
    _test_all_to_all(dtype)

    _test_barrier()
    _test_broadcast_object()

    cleanup()


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="MCCL not released for qy1 dev3.0.0"
)
@testing.skip_if_not_multiple_musa_device
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.half, torch.bfloat16],
)
def test_dist_comm(dtype):
    # Always use 2-cards for test
    _test_demo(_test_function, 2, dtype)


@pytest.mark.skipif(testing.get_musa_arch() < 31, reason="MCCL not released for ph1")
@testing.skip_if_not_multiple_musa_device
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e5m2, torch.float8_e4m3fn],
)
def test_dist_comm_fp8(dtype):
    # Always use 2-cards for test
    _test_demo(_test_function, 2, dtype)


if __name__ == "__main__":
    _test_demo(_test_function, 2, torch.float32)
