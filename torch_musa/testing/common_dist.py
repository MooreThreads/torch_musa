"""Distributed test utilities."""

from functools import wraps

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    run_subtests as _run_subtests,
)


def skip_if_lt_x_gpu(x):
    return pytest.mark.skipif(
        torch.musa.device_count() < x, reason=f"Need at least {x} MUSA devices"
    )


class MultiProcessingTest(MultiProcessTestCase):
    """PyTorch distributed multiprocessing harness for torch_musa tests."""

    @property
    def world_size(self):
        return min(torch.musa.device_count(), 8)

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def backend(self) -> str:
        return "mccl" if torch.musa.is_available() else "gloo"

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def bootstrap_default_process_group(self) -> None:
        if torch.musa.is_available() and torch.musa.device_count():
            torch.musa.set_device(self.rank % torch.musa.device_count())

        dist.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=int(self.world_size),
            init_method=f"file://{self.file_name}",
        )

    def run_test(self, test_name: str, parent_pipe) -> None:
        original_test = getattr(self, test_name)

        @wraps(original_test)
        def wrapped_test() -> None:
            self.bootstrap_default_process_group()
            dist.barrier()
            original_test()
            dist.barrier()

        setattr(self, test_name, wrapped_test)
        try:
            super().run_test(test_name, parent_pipe)
        finally:
            setattr(self, test_name, original_test)

    def run_subtests(self, *args, **kwargs):
        return _run_subtests(self, *args, **kwargs)


def run_subtests(cls_inst, subtest_config, test_fn, *test_args, **test_kwargs):
    return _run_subtests(
        cls_inst,
        subtest_config,
        test_fn,
        *test_args,
        **test_kwargs,
    )
