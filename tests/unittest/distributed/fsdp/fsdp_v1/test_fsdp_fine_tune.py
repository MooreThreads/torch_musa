"""Test fsdp finetune"""

# pylint: disable=unused-argument,invalid-name

from typing import (
    Optional,
    Iterable,
    Callable,
    Union,
    List,
)
import itertools
import pytest
import torch
from torch import nn

import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.wrap import _Policy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)

import torch_musa
from torch_musa import testing
from torch_musa.testing.common_fsdp import FSDPTest, skip_if_lt_x_gpu
from torch_musa.distributed.fsdp import ShardedGradScaler

NUM_DEVICES_FOR_TESTING = 4
HYBRID_SHARD_STRATEGY = [
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
]


class FSDPConfig:
    """class FSDPConfig"""

    def __init__(self) -> None:
        self.sharding_strategy: Optional[ShardingStrategy] = None
        self.cpu_offload: Optional[CPUOffload] = None
        self.auto_wrap_policy: Optional[Union[Callable, _Policy]] = None
        self.backward_prefetch: Optional[BackwardPrefetch] = (
            BackwardPrefetch.BACKWARD_PRE
        )
        self.mixed_precision: Optional[MixedPrecision] = None
        self.ignored_modules: Optional[Iterable[torch.nn.Module]] = None
        self.param_init_fn: Optional[Callable[[torch.nn.Module], None]]
        self.device_id: Optional[Union[int, torch.device]] = None
        self.sync_module_states: bool = False
        self.forward_prefetch: bool = False
        self.limit_all_gathers: bool = True
        self.use_orig_params: bool = False
        self.ignored_states: Union[
            Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]
        ] = None
        self.device_mesh: Optional[DeviceMesh] = None

    def __repr__(self) -> str:
        msg = "FSDP Config:[\n"
        for k, v in self.__dict__.items():
            msg += f"    {k}: {v}\n"
        msg += "]"
        return msg


def wrap_policies() -> List[_Policy]:
    def true_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        return True

    def false_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        return False

    return [None, true_policy, false_policy]


def supported_full_sharding_strategies() -> List[ShardingStrategy]:
    return [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP]


def supported_hybrid_sharding_strategies() -> List[ShardingStrategy]:
    return [ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2]


def supported_cpu_offload_configs() -> List[CPUOffload]:
    return [None, CPUOffload(offload_params=True), CPUOffload(offload_params=False)]


def supported_backward_prefetch_configs() -> List[BackwardPrefetch]:
    if torch.musa.core._utils._get_musa_arch() < 31:
        return [
            None,
        ]

    return [None, BackwardPrefetch.BACKWARD_PRE, BackwardPrefetch.BACKWARD_POST]


def supported_mixed_precision_configs() -> List[MixedPrecision]:
    return [None, MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)]


def supported_fsdp_configs():
    """generator of fsdp_configs"""

    def named_product(**items):
        keys = list(items.keys())
        values = list(items.values())
        for _values in itertools.product(*values):
            yield dict(zip(keys, _values))

    for params in named_product(
        cpu_offload=supported_cpu_offload_configs(),
        # auto_wrap_policy=wrap_policies(),
        backward_prefetch=supported_backward_prefetch_configs(),
        mixed_precision=supported_mixed_precision_configs(),
        sync_module_states=[
            False,
        ],
        forward_prefetch=[
            False,
        ],
        limit_all_gathers=[True, False],
        use_orig_params=[
            False,
        ],
    ):
        fsdp_config = FSDPConfig()
        for k, v in params.items():
            setattr(fsdp_config, k, v)
        yield fsdp_config


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="Skip on arch's version older than 22"
)
class TestFSDPFineTune(FSDPTest):
    """class TestFSDPFineTune"""

    @property
    def world_size(self) -> int:
        # we will also test hybrid sharding
        return min(torch.musa.device_count(), NUM_DEVICES_FOR_TESTING)

    def _init_seq_module(self) -> nn.Module:
        modules = []
        for _ in range(2):
            modules += [nn.Linear(6, 6), nn.ReLU()]
        seq = nn.Sequential(*modules)
        return seq

    def _device_mesh_plan(self, device="musa", hybrid_shard=False):
        if hybrid_shard:
            assert (self.world_size >= NUM_DEVICES_FOR_TESTING) and (
                self.world_size % 2 == 0
            )
            replica_group_size = 2
            sharding_group_size = self.world_size // 2
            device_mesh = init_device_mesh(
                device, (replica_group_size, sharding_group_size)
            )
        else:
            device_mesh = init_device_mesh(device, (self.world_size, 1))
        return device_mesh

    @skip_if_lt_x_gpu(NUM_DEVICES_FOR_TESTING)
    def test_fsdp_full_sharding_training(self):
        for sharding_strategy in supported_full_sharding_strategies():
            for fsdp_config in supported_fsdp_configs():
                setattr(fsdp_config, "sharding_strategy", sharding_strategy)
                self._test_fsdp_basic_training(fsdp_config)

    @skip_if_lt_x_gpu(NUM_DEVICES_FOR_TESTING)
    def test_fsdp_hybrid_sharding_training(self):
        for sharding_strategy in supported_hybrid_sharding_strategies():
            for fsdp_config in supported_fsdp_configs():
                setattr(fsdp_config, "sharding_strategy", sharding_strategy)
                self._test_fsdp_basic_training(fsdp_config)

    def _test_fsdp_basic_training(self, fsdp_config):
        device_mesh = self._device_mesh_plan(
            hybrid_shard=fsdp_config.sharding_strategy in HYBRID_SHARD_STRATEGY
        )
        setattr(fsdp_config, "device_mesh", device_mesh)
        if self.rank == 0:
            print(fsdp_config)

        device = torch_musa.current_device()

        model = self._init_seq_module().to(device)
        fsdp_model = FSDP(
            module=model,
            sharding_strategy=fsdp_config.sharding_strategy,
            cpu_offload=fsdp_config.cpu_offload,
            backward_prefetch=fsdp_config.backward_prefetch,
            mixed_precision=fsdp_config.mixed_precision,
            sync_module_states=fsdp_config.sync_module_states,
            forward_prefetch=fsdp_config.forward_prefetch,
            limit_all_gathers=fsdp_config.limit_all_gathers,
            use_orig_params=fsdp_config.use_orig_params,
            ignored_states=fsdp_config.ignored_states,
            device_mesh=fsdp_config.device_mesh,
        )

        scaler = ShardedGradScaler(growth_interval=1)
        optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.001)
        inp = torch.randn((2, 6), device=device)
        for _ in range(5):
            out = fsdp_model(inp)
            loss = out.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # make sure models are running on MUSA
        assert loss.device.type == "musa"
        dist.barrier()

        if self.rank == 0:
            print("Test passed")
