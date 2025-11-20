"""Test save and load FSDP model's state_dict"""

# pylint: disable=invalid-name,missing-function-docstring,C0301

from typing import Dict
from contextlib import nullcontext
import copy

import torch
from torch import nn
from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
)
from torch.testing._internal.common_fsdp import MLP

from torch_musa.testing.common_fsdp import FSDPTest, skip_if_lt_x_gpu


NUM_GPUS_FOR_TEST_FSDP_SD = 4


@skip_if_lt_x_gpu(NUM_GPUS_FOR_TEST_FSDP_SD)
class TestFullyShardStateDict(FSDPTest):
    """Test class for testing the correctness of loading and saving state_dict of FSDP model"""

    @property
    def world_size(self) -> int:
        return NUM_GPUS_FOR_TEST_FSDP_SD

    def test_dp_state_dict_save_load(self):
        fsdp_mesh = self._device_mesh_plan(hybrid_shard=False)
        self.run_subtests(
            {"mesh": [fsdp_mesh]},
            self._test_dp_state_dict_save_load,
        )
        if self.world_size % 2 != 0:
            return
        hsdp_mesh = self._device_mesh_plan(hybrid_shard=True)
        self.run_subtests(
            {"mesh": [hsdp_mesh]},
            self._test_dp_state_dict_save_load,
        )

    def test_dp_state_dict_cpu_offload(self):
        self.run_subtests(
            {
                "offload_policy": [
                    CPUOffloadPolicy(pin_memory=True),
                    CPUOffloadPolicy(pin_memory=False),
                ],
                "cpu_state_dict": [True, False],
            },
            self._test_dp_state_dict_cpu_offload,
        )

    def _test_dp_state_dict_cpu_offload(
        self, offload_policy: CPUOffloadPolicy, cpu_state_dict: bool
    ):
        mlp_dim = 4
        torch.manual_seed(42)
        with torch.device("meta"):
            model = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim, bias=False),
                nn.Linear(mlp_dim, mlp_dim, bias=False),
            )
        for module in model:
            fully_shard(module, offload_policy=offload_policy)
        fully_shard(model, offload_policy=offload_policy)

        # split full sd into multiple pieces
        # to test loading with `strict=False`
        state_dicts = []
        for name, dtensor in model.named_parameters():
            full_tensor = torch.randn(dtensor.size())
            sharded_tensor = distribute_tensor(
                full_tensor, dtensor.device_mesh, dtensor.placements
            )
            if cpu_state_dict:
                sharded_tensor = sharded_tensor.cpu()
            state_dicts.append({name: sharded_tensor})

        # check that we can load with some parameters still on meta device
        for sd in state_dicts:
            model.load_state_dict(sd, assign=True, strict=False)

        # lazy init without error
        inp = torch.rand((mlp_dim, mlp_dim), device="musa")

        context = (
            self.assertRaisesRegex(
                RuntimeError,
                "Found following parameters on non-CPU device: "
                r"\[\('0.weight', device\(type='musa'",
            )
            if not cpu_state_dict
            else nullcontext()
        )
        with context:
            model(inp).sum()
            state_dict = model.state_dict()
            for name, dtensor in state_dict.items():
                self.assertEqual(dtensor.device.type, "cpu")

    def _test_dp_state_dict_save_load(self, mesh: DeviceMesh):
        torch.manual_seed(42)
        mlp_dim = 16
        base_model = nn.Sequential(
            MLP(mlp_dim),
            nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),
            MLP(mlp_dim),
        )
        # check reshard_after_forward=True
        model1 = copy.deepcopy(base_model)
        for m in model1:
            fully_shard(m, mesh=mesh)
        fully_shard(model1, mesh=mesh)
        self._test_state_dict_save_load(model1)

        # check `reshard_after_forward=False` before and after a forward
        model2 = copy.deepcopy(base_model)
        for m in model2:
            fully_shard(m, mesh=mesh, reshard_after_forward=False)
        fully_shard(model2, mesh=mesh, reshard_after_forward=False)
        self._test_state_dict_save_load(model2)
        ref_sharded_sd = model2.state_dict()
        inp = torch.randn((2, mlp_dim), device="musa")
        model2(inp)
        sharded_sd = model2.state_dict()
        for x, y in zip(set(ref_sharded_sd.keys()), set(sharded_sd.keys())):
            assert x == y
        for key, value in ref_sharded_sd.items():
            assert torch.allclose(value.to_local(), sharded_sd[key].to_local())

    def test_dp_tp_state_dict_save_load(self):
        dp_size = 2
        mesh = init_device_mesh(
            "musa", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        self.run_subtests(
            {
                "mesh": [
                    mesh,
                ]
            },
            self._test_dp_tp_state_dict_save_load,
        )

    def _test_dp_tp_state_dict_save_load(self, mesh: DeviceMesh):
        dp_mesh, tp_mesh = mesh["dp"], mesh["tp"]
        mlp_dim = 8
        torch.manual_seed(42)
        model = nn.Sequential(*[MLP(mlp_dim) for _ in range(2)])
        # apply tp
        model = parallelize_module(
            model,
            device_mesh=tp_mesh,
            parallelize_plan={
                "0.in_proj": ColwiseParallel(),
                "0.out_proj": RowwiseParallel(),
                "1.in_proj": ColwiseParallel(),
                "1.out_proj": RowwiseParallel(),
            },
        )
        # apply dp
        for m in model:
            fully_shard(m, mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)
        self._test_state_dict_save_load(model)

    def _test_state_dict_save_load(self, model: nn.Module):
        for name, param in model.named_parameters():
            assert isinstance(
                param, DTensor
            ), f"Expects parameters to be sharded as DTensors but got {name} as {type(param)}"

        old_fill_value = 1
        new_fill_value = 42 + self.rank
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(old_fill_value)

        param_name_to_data_ptr = {
            n: p.to_local().data_ptr() for n, p in model.named_parameters()
        }

        state_dict = model.state_dict()

        # Verify that keys match, values are DTensors, and values share the
        # same storage as the existing sharded parameter data
        for k0, k1 in zip(set(state_dict.keys()), set(param_name_to_data_ptr.keys())):
            assert k0 == k1

        for p_name, tensor in state_dict.items():
            assert isinstance(tensor, DTensor)
            if param_name_to_data_ptr[p_name] == 0:
                assert self.rank > 0
                assert torch.count_nonzero(tensor.to_local()).item() == 0
            else:
                assert tensor.to_local().data_ptr() == param_name_to_data_ptr[p_name]

        # Verify that we can load a new state dict that contains DTensors with
        # storages different from the current model parameters
        new_state_dict: Dict[str, DTensor] = {}
        for param_name, dtensor in state_dict.items():
            new_state_dict[param_name] = dtensor.detach().clone().fill_(new_fill_value)
        for param in model.parameters():
            assert torch.allclose(
                param.to_local(), torch.ones_like(param.to_local()) * old_fill_value
            )
        model.load_state_dict(new_state_dict)
        for param in model.parameters():
            local_param = param.to_local()
            assert torch.allclose(
                local_param, torch.ones_like(local_param) * new_fill_value
            )
