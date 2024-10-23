"""utilities for testing FSDP"""

# pylint: disable=invalid-name,
import os
import sys
import traceback
import time
from functools import wraps
import types
from typing import List, Dict, Callable, Any, Tuple
import itertools
import unittest
import pytest

import torch
import torch.distributed as dist
import torch_musa


def skip_if_lt_x_gpu(x):
    return pytest.mark.skipif(
        torch_musa.device_count() < x, reason=f"Need at least {x} MUSA devices"
    )


class FSDPTest(unittest.TestCase):
    """class FSDPTest

    borrowed from PyTorch's testing code, but some modifications have been made to work with pytest
    """

    MAIN_PROCESS_RANK = -1

    TEST_ERROR_EXIT_CODE = 10

    def _join_processes(self):
        subprocess_error = False
        while True:
            for _, p in enumerate(self.processes):
                if p.exitcode == FSDPTest.TEST_ERROR_EXIT_CODE:
                    active_children = torch.multiprocessing.active_children()
                    for ac in active_children:
                        ac.terminate()
                    subprocess_error = True
                    break
            if subprocess_error:
                break
            if all(p.exitcode is not None for p in self.processes):
                break
            time.sleep(0.1)

        # To make pytest behave normally, an assert was added here
        assert not subprocess_error

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes()
            else:
                fn()

        return types.MethodType(wrapper, self)

    def runTest(self):
        # NB: do not remove this non-op function if using pytest,
        # this will add an extra(non-op) test case though.
        pass

    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def _random_port(self):
        return int(time.time() // 256 % (32768 - 10000) + 10000)

    def setUp(self) -> None:
        super().setUp()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self._random_port())
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK

        if self._testMethodName != "runTest":
            self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()
        if self._testMethodName != "runTest":
            for p in self.processes:
                p.terminate()

            # reset the processes
            self.processes = []

    @property
    def world_size(self):
        return min(torch_musa.device_count(), 8)

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    def _current_test_name(self) -> str:
        return self.id().split(".")[-1]

    def _start_processes(self, proc) -> None:
        self.processes = []
        for rank in range(int(self.world_size)):
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(rank, self._current_test_name()),
            )
            process.start()
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    def run_test(self, test_name: str) -> None:
        try:
            getattr(self, test_name)()
        except Exception as _:  # pylint: disable=broad-exception-caught
            print(
                f"Caught execption: \n{traceback.format_exc()} exiting "
                f"process {self.rank} with exit code: {FSDPTest.TEST_ERROR_EXIT_CODE}"
            )
            sys.exit(FSDPTest.TEST_ERROR_EXIT_CODE)

    @classmethod
    def _run(cls, rank: int, test_name: str) -> None:
        self = cls(test_name)
        self.rank = rank  # 0 ~ N

        backend = "mccl" if torch_musa.is_available() else "gloo"
        dist.init_process_group(
            backend=backend, rank=self.rank, world_size=int(self.world_size)
        )

        if torch_musa.is_available() and torch_musa.device_count():
            torch_musa.set_device(self.rank % torch_musa.device_count())

        dist.barrier()
        self.run_test(test_name)
        dist.barrier()
        dist.destroy_process_group()

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)


def run_subtests(
    cls_inst: Any,
    subtest_config: Dict[str, List[Any]],
    test_fn: Callable,
    *test_args,
    **test_kwargs: Any,
):
    """
    Runs a test function given by ``test_fn`` as a subtest according to the
    configurations specified by ``subtest_config``. This amortizes the
    costly setup overhead (including process spawn and initializing the
    process group) over the subtests.

    Args:
        cls_inst: class instance.
        subtest_config (Dict[str, List[Any]]): A mapping from subtest
            keyword argument name to a list of its possible values.
        test_fn (Callable): A callable that runs the actual test.
        test_args: Positional arguments to pass to ``test_fn``.
        test_kwargs: Keyword arguments to pass to ``test_fn``.
    """
    # Convert the config mapping to a list to have a fixed order
    subtest_config_items: List[Tuple[str, List[Any]]] = list(subtest_config.items())
    subtest_config_keys: List[str] = [item[0] for item in subtest_config_items]
    subtest_config_values: List[List[Any]] = [item[1] for item in subtest_config_items]
    for values in itertools.product(*subtest_config_values):
        # Map keyword to chosen value
        subtest_kwargs = dict(zip(subtest_config_keys, values))
        with cls_inst.subTest(**subtest_kwargs):
            if dist.get_rank() == 0:
                print(f"Current test case's param: {subtest_kwargs}")
            test_fn(*test_args, **test_kwargs, **subtest_kwargs)
        dist.barrier()
