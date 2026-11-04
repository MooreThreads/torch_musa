"""Test FSDP2 communication types and overlap levels."""

# pylint: disable=all
import os
import gc
import sys
from contextlib import contextmanager

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama3_model import (
    LLAMA3_TEST_BATCH_SIZE,
    LLAMA3_TEST_LAYERS,
    LLAMA3_TEST_SEQ_LEN,
    LLAMA3_VOCAB_SIZE,
    init_llama3_model,
)

from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
from torch.distributed._tensor import DTensor
from torch.testing._internal.common_distributed import TIMEOUT_OVERRIDE
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from torch_musa.distributed._composable.fsdp import custom_overlap_patch
from torch_musa.testing import get_musa_arch
from torch_musa.testing.common_fsdp import FSDPTest, skip_if_lt_x_gpu

# Each parametrized Llama3 FSDP communication test may exceed the default 300-second timeout.
TIMEOUT_OVERRIDE.update(
    {
        "test_comm_type_and_overlap_level_against_baseline_hybrid_shard_False": 600,
        "test_comm_type_and_overlap_level_against_baseline_hybrid_shard_True": 600,
    }
)

NUM_DEVICES_FOR_TESTING = 4
COMM_TYPE_ENV = "TORCH_MUSA_FSDP2_COMM_TYPE"
OVERLAP_LEVEL_ENV = "TORCH_MUSA_FSDP2_OVERLAP_LEVEL"
DISABLE_OVERLAP_ENV = "TORCH_MUSA_FSDP2_DISABLE_OVERLAP"
FSDP2_ENV_NAMES = (
    COMM_TYPE_ENV,
    OVERLAP_LEVEL_ENV,
    DISABLE_OVERLAP_ENV,
)


@skip_if_lt_x_gpu(NUM_DEVICES_FOR_TESTING)
@pytest.mark.skipif(get_musa_arch() < 31, reason="requires MUSA arch >= mp_31")
@instantiate_parametrized_tests
class TestFSDP2CommOverlap(FSDPTest):
    """Testing FSDP2 with different overlap settings"""

    @property
    def world_size(self) -> int:
        return NUM_DEVICES_FOR_TESTING

    def test_environment_configuration_is_dynamic(self) -> None:
        configurations = (
            ({COMM_TYPE_ENV: "0", OVERLAP_LEVEL_ENV: "0"}, 0, 0),
            ({COMM_TYPE_ENV: "2", OVERLAP_LEVEL_ENV: "3"}, 2, 3),
        )
        for environment, expected_comm_type, expected_overlap_level in configurations:
            with _fsdp2_environment(**environment):
                assert custom_overlap_patch._resolved_comm_type() == expected_comm_type
                comm_ctx = FSDPCommContext()
                custom_overlap_patch.comm_context_lazy_init(
                    comm_ctx, torch.device("musa")
                )
                assert (
                    custom_overlap_patch._get_fsdp2_overlap_level().value
                    == expected_overlap_level
                )
                _assert_stream_topology(
                    _stream_topology(comm_ctx), expected_overlap_level
                )

    @parametrize("hybrid_shard", (False, True))
    def test_comm_type_and_overlap_level_against_baseline(
        self, hybrid_shard: bool
    ) -> None:
        mesh = self._device_mesh_plan(hybrid_shard=hybrid_shard)
        baseline = self._run_training(mesh, comm_type=0, overlap_level=0)
        self.run_subtests(
            {
                "comm_type": (1, 2),
                "overlap_level": (0, 1, 2, 3),
            },
            self._test_comm_type_and_overlap_level_against_baseline,
            baseline,
            mesh,
            hybrid_shard,
        )

    def _test_comm_type_and_overlap_level_against_baseline(
        self,
        baseline: dict,
        mesh,
        hybrid_shard: bool,
        comm_type: int,
        overlap_level: int,
    ) -> None:
        print(
            f"Testing comm_type: {comm_type}, overlap_level: {overlap_level}, hybrid_shard: {hybrid_shard}",
            flush=True,
        )
        output = self._run_training(mesh, comm_type, overlap_level)
        assert output["resolved_overlap_level"] == overlap_level
        _assert_comm_type(output["reduce_scatter_comm_types"], comm_type)
        _assert_stream_topology(output["stream_topology"], overlap_level)

        torch.testing.assert_close(
            output["loss"],
            baseline["loss"],
            rtol=1e-3,
            atol=1e-4,
            msg=f"forward loss mismatch on rank {self.rank}, output: {output['loss']}, baseline: {baseline['loss']}",
        )

    def _run_training(
        self,
        mesh,
        comm_type: int | None = None,
        overlap_level: int | None = None,
    ) -> dict:
        environment = {}
        if comm_type is not None:
            environment[COMM_TYPE_ENV] = str(comm_type)
        if overlap_level is not None:
            environment[OVERLAP_LEVEL_ENV] = str(overlap_level)

        with _fsdp2_environment(**environment):
            torch.manual_seed(42)
            model = init_llama3_model(LLAMA3_TEST_LAYERS, LLAMA3_TEST_SEQ_LEN)
            init_device = torch.device("musa", torch.musa.current_device())
            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights()
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
            for layer in model.layers:
                fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
            fully_shard(model, mesh=mesh, mp_policy=mp_policy)

            optim = torch.optim.SGD(model.parameters(), lr=1e-2)
            torch.manual_seed(43 + self.rank)
            for _ in range(10):
                optim.zero_grad()
                inp = torch.randint(
                    LLAMA3_VOCAB_SIZE,
                    (LLAMA3_TEST_BATCH_SIZE, LLAMA3_TEST_SEQ_LEN),
                    device=torch.musa.current_device(),
                )
                loss = model(inp).float().mean()
                loss.backward()
                optim.step()
            torch.musa.synchronize()

            root_state = model._get_fsdp_state()
            comm_ctx = root_state._comm_ctx
            reduce_scatter_comm_types = []
            for module in model.modules():
                if not isinstance(module, FSDPModule):
                    continue
                param_group = module._get_fsdp_state()._fsdp_param_group
                assert param_group is not None
                reduce_scatter_comm_types.append(
                    type(param_group._reduce_scatter_comm).__name__
                )

            loss = loss.detach().cpu()

            del model, optim
            gc.collect()
            torch.musa.empty_cache()

            return {
                "loss": loss,
                "resolved_overlap_level": (
                    custom_overlap_patch._get_fsdp2_overlap_level().value
                ),
                "reduce_scatter_comm_types": reduce_scatter_comm_types,
                "stream_topology": _stream_topology(comm_ctx),
            }


@contextmanager
def _fsdp2_environment(**environment):
    original_environment = {name: os.environ.get(name) for name in FSDP2_ENV_NAMES}
    try:
        for name in FSDP2_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(environment)
        yield
    finally:
        for name in FSDP2_ENV_NAMES:
            os.environ.pop(name, None)
        for name, value in original_environment.items():
            if value is not None:
                os.environ[name] = value


def _assert_comm_type(comm_class_names: list[str], comm_type: int) -> None:
    if comm_type == 0:
        assert all(
            "IntraNodeLowContentionCommReduceScatter" not in name
            and "ProcessGroupMCCLSymmMemReduceScatter" not in name
            for name in comm_class_names
        )
    elif comm_type == 1:
        assert set(comm_class_names) == {"IntraNodeLowContentionCommReduceScatter"}
    else:
        assert set(comm_class_names) == {"ProcessGroupMCCLSymmMemReduceScatter"}


def _assert_stream_topology(topology: dict[str, bool], overlap_level: int) -> None:
    expected = {
        0: {
            "copy_in_is_all_gather": True,
            "copy_in_is_reduce_scatter": True,
            "copy_in_is_all_reduce": True,
            "all_gather_is_reduce_scatter": True,
            "all_gather_is_all_reduce": True,
            "reduce_scatter_is_all_reduce": True,
        },
        1: {
            "copy_in_is_all_gather": False,
            "copy_in_is_reduce_scatter": False,
            "copy_in_is_all_reduce": True,
            "all_gather_is_reduce_scatter": True,
            "all_gather_is_all_reduce": False,
            "reduce_scatter_is_all_reduce": False,
        },
        2: {
            "copy_in_is_all_gather": True,
            "copy_in_is_reduce_scatter": True,
            "copy_in_is_all_reduce": False,
            "all_gather_is_reduce_scatter": True,
            "all_gather_is_all_reduce": False,
            "reduce_scatter_is_all_reduce": False,
        },
        3: {
            "copy_in_is_all_gather": False,
            "copy_in_is_reduce_scatter": False,
            "copy_in_is_all_reduce": False,
            "all_gather_is_reduce_scatter": True,
            "all_gather_is_all_reduce": False,
            "reduce_scatter_is_all_reduce": False,
        },
        4: {
            "copy_in_is_all_gather": False,
            "copy_in_is_reduce_scatter": False,
            "copy_in_is_all_reduce": False,
            "all_gather_is_reduce_scatter": False,
            "all_gather_is_all_reduce": False,
            "reduce_scatter_is_all_reduce": False,
        },
    }
    assert topology == expected[overlap_level]


def _stream_topology(comm_ctx) -> dict[str, bool]:
    copy_in_stream = comm_ctx.all_gather_copy_in_stream
    all_gather_stream = comm_ctx.all_gather_stream
    reduce_scatter_stream = comm_ctx.reduce_scatter_stream
    all_reduce_stream = comm_ctx.all_reduce_stream
    return {
        "copy_in_is_all_gather": copy_in_stream is all_gather_stream,
        "copy_in_is_reduce_scatter": copy_in_stream is reduce_scatter_stream,
        "copy_in_is_all_reduce": copy_in_stream is all_reduce_stream,
        "all_gather_is_reduce_scatter": all_gather_stream is reduce_scatter_stream,
        "all_gather_is_all_reduce": all_gather_stream is all_reduce_stream,
        "reduce_scatter_is_all_reduce": reduce_scatter_stream is all_reduce_stream,
    }

