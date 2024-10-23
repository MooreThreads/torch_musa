"""Demo test of DistributedDataParall"""

# pylint: disable = W0611, C0103, C0116
import os
import torch
import pytest
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_musa
from torch_musa import testing


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(x)


def start(rank, world_size):
    ip, port = testing.gen_ip_port()
    if os.getenv("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = ip
    if os.getenv("MASTER_PORT") is None:
        os.environ["MASTER_PORT"] = port
    dist.init_process_group("mccl", rank=rank, world_size=world_size)


def clean():
    dist.destroy_process_group()


def runner(rank, world_size):
    torch_musa.set_device(rank)
    start(rank, world_size)
    model = Model().to("musa")
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for _ in range(5):
        input_tensor = torch.randn(5, dtype=torch.float, requires_grad=True).to("musa")
        target_tensor = torch.zeros(5, dtype=torch.float).to("musa")
        output_tensor = ddp_model(input_tensor)
        loss_f = nn.MSELoss()
        loss = loss_f(output_tensor, target_tensor)
        loss.backward()
        optimizer.step()
    clean()


def train(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="MCCL not released for qy1 dev3.0.0"
)
@testing.skip_if_not_multiple_musa_device
def test_DistributedDataParallel():
    if torch.musa.device_count() > 2:
        train(runner, 2)
    else:
        train(runner, torch.musa.device_count())


if __name__ == "__main__":
    train(runner, 2)
