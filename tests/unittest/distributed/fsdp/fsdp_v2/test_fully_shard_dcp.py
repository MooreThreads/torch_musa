"""Test Distributed Checkpoint save and load function"""

# pylint: disable=invalid-name,missing-function-docstring
import os
import io
import copy
import shutil

import torch
from torch import nn

from torch.distributed._composable.fsdp import fully_shard
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from torch.distributed._state_dict_utils import (
    _check_state_dict_similarity,
    _copy_state_dict,
    _create_cpu_state_dict
)

from torch.testing._internal.common_fsdp import MLP
from torch_musa.testing.common_fsdp import FSDPTest, skip_if_lt_x_gpu

NUM_GPUS_FOR_TEST_FSDP_SD = 4


def delete_folder(folder_name):
    """Delete the specified folder in the current path"""
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # get the complete path
    folder_path = os.path.join(current_dir, folder_name)

    try:
        if not os.path.exists(folder_path):
            print(f"{folder_name} unexist")
            return

        if not os.path.isdir(folder_path):
            print(f"error: '{folder_name}' is not a directory")
            return

        shutil.rmtree(folder_path)
        print(f"succeeded delete directory: {folder_path}")

    except PermissionError:
        print(f"error: permission denied to delete: '{folder_name}'")
    except Exception as e:  # pylint: disable=broad-except
        print(f"An error occurred while deleting the folder: {str(e)}")


@skip_if_lt_x_gpu(NUM_GPUS_FOR_TEST_FSDP_SD)
class TestFullyShardD(FSDPTest):
    """Test class for testing the correctness of loading and saving state_dict of FSDP model"""

    @property
    def world_size(self) -> int:
        return NUM_GPUS_FOR_TEST_FSDP_SD

    def _test_dp_shard_dcp_save_load(self, async_save=False):
        if self.rank == 0:
            delete_folder("checkpoint")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        state_dict_dir = os.path.join(current_dir, "checkpoint")

        fsdp_mesh = self._device_mesh_plan(hybrid_shard=False)
        mlp_dim = 16
        base_model = nn.Sequential(
            MLP(mlp_dim),
            nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),
            MLP(mlp_dim),
        )

        model1 = copy.deepcopy(base_model)
        model2 = copy.deepcopy(base_model)

        for m in model1:
            fully_shard(m, mesh=fsdp_mesh)
        fully_shard(model1, mesh=fsdp_mesh)

        for m in model2:
            fully_shard(m, mesh=fsdp_mesh)
        fully_shard(model2, mesh=fsdp_mesh)

        optimizer = torch.optim.AdamW(model1.parameters(), lr=0.1)
        dst_optimizer = torch.optim.AdamW(model2.parameters(), lr=0.1)
        optimizer.zero_grad()
        inp = torch.randn((2, mlp_dim), device="musa")
        out = model1(inp).sum()
        out.backward()
        optimizer.step()

        model_state_dict, optim_state_dict = get_state_dict(model1, optimizer)
        state_dict = {"model": model_state_dict, "optimizer": optim_state_dict}
        if not async_save:
            dcp.save(state_dict, checkpoint_id=state_dict_dir)
        else:
            pg = torch.distributed.new_group(backend="gloo")
            async_future = dcp.async_save(
                state_dict, checkpoint_id=state_dict_dir, process_group=pg
            )
            async_future.result()

        loacl_file_path = os.path.join(state_dict_dir, f"__{self.rank}_0.distcp")
        assert os.path.exists(loacl_file_path)

        dst_model_state_dict, dst_optim_state_dict = get_state_dict(
            model2, dst_optimizer
        )
        dst_state_dict = {
            "model": dst_model_state_dict,
            "optimizer": dst_optim_state_dict,
        }

        dcp.load(dst_state_dict, checkpoint_id=state_dict_dir)
        set_state_dict(
            model2,
            dst_optimizer,
            model_state_dict=dst_state_dict["model"],
            optim_state_dict=dst_state_dict["optimizer"],
        )

        torch.allclose(
            model1[0].in_proj.weight.to_local(), model2[0].in_proj.weight.to_local()
        )

        optim_exp_avg = list(optimizer.state.values())[0]["exp_avg"]
        dst_optim_exp_avg = list(dst_optimizer.state.values())[0]["exp_avg"]
        optim_exp_avg_sq = list(optimizer.state.values())[0]["exp_avg_sq"]
        dst_optim_exp_avg_sq = list(dst_optimizer.state.values())[0]["exp_avg_sq"]

        torch.allclose(optim_exp_avg.to_local(), dst_optim_exp_avg.to_local())
        torch.allclose(optim_exp_avg_sq.to_local(), dst_optim_exp_avg_sq.to_local())

        if self.rank == 0:
            delete_folder("checkpoint")


    def test_dp_shard_dcp_save_load(self):
        self._test_dp_shard_dcp_save_load(False)
        self._test_dp_shard_dcp_save_load(True)


    def test_create_cpu_state_dict(self):
        device = torch.device("musa")
        buffer = io.BytesIO()
        torch.save(torch.ones(10), buffer)
        buffer.seek(0)
        state_dict = {
            "tensor1": torch.arange(10, device=device),
            "tensor2": torch.ones(10, device=device),
            "non_tensor_bytes_io": copy.deepcopy(buffer),
            "non_tensor_bytes": buffer.read(),
            "step": torch.tensor(7, dtype=torch.float),
            "lr": 1.5,
            "nested": {"list": [1, 2, 3, 4]},
        }

        def _verify(cpu_state_dict):
            # Verify the correctness of _check_state_dict_similarity()
            self.assertTrue(_check_state_dict_similarity(state_dict, cpu_state_dict))
            tensor1 = cpu_state_dict["tensor1"]
            cpu_state_dict["tensor1"] = torch.arange(11)
            self.assertFalse(_check_state_dict_similarity(state_dict, cpu_state_dict))
            cpu_state_dict["tensor1"] = tensor1

            _copy_state_dict(state_dict, cpu_state_dict)

            # Verify if _copy_state_dict works
            for v in cpu_state_dict.values():
                if isinstance(v, torch.Tensor):
                    self.assertFalse(v.is_musa)
            assert (cpu_state_dict["tensor1"] == torch.arange(10)).all()
            assert (cpu_state_dict["tensor2"] == torch.ones(10)).all()
            buffer.seek(0)
            cpu_state_dict["non_tensor_bytes_io"].seek(0)
            self.assertEqual(
                cpu_state_dict["non_tensor_bytes_io"].read(), buffer.read()
            )
            buffer.seek(0)
            self.assertEqual(cpu_state_dict["non_tensor_bytes"], buffer.read())
            self.assertEqual(cpu_state_dict["lr"], 1.5)
            self.assertEqual(cpu_state_dict["step"], 7)
            self.assertEqual(cpu_state_dict["nested"], {"list": [1, 2, 3, 4]})

        cpu_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True)
        _verify(cpu_state_dict)
        cpu_state_dict = _create_cpu_state_dict(state_dict, share_memory=True)
        _verify(cpu_state_dict)
        cpu_state_dict = _create_cpu_state_dict(
            state_dict, share_memory=True, pin_memory=True
        )
        _verify(cpu_state_dict)
