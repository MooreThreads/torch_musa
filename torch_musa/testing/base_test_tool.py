"""Some basic functions for tests."""
# pylint: disable=too-few-public-methods, too-many-locals, too-many-statements, consider-using-dict-items
# pylint: disable=unused-import, too-many-branches, not-callable, missing-function-docstring

from contextlib import ExitStack, nullcontext
import random
from typing import Callable
from functools import wraps

import time
import types
import pytest
import numpy as np
import torch
import torch_musa


MUSA_AVAILABLE = torch_musa.is_available()
MULTIGPU_AVAILABLE = MUSA_AVAILABLE and torch_musa.device_count() >= 2

skip_if_musa_unavailable = pytest.mark.skipif(
    not MUSA_AVAILABLE, reason="No MUSA device is detected"
)
skip_if_not_multiple_musa_device = pytest.mark.skipif(
    not MULTIGPU_AVAILABLE, reason="Expect multiple MUSA devices"
)


def test_on_nonzero_card_if_multiple_musa_device(musa_device: int):
    """
    Decorator for conducting operators' test on nonzero card.
    """

    def wrapper(test_func):
        @wraps(test_func)
        def infunc(*args, **kwargs):
            if MULTIGPU_AVAILABLE:
                with torch_musa.device(musa_device):
                    print(f"testing on card {musa_device}...")
                    test_func(*args, **kwargs)
            test_func(*args, **kwargs)

        return infunc

    return wrapper


def get_raw_data():
    return [
        torch.randn(10),
        # to test non_contiguous tensor(storage_offset != 0),
        torch.randn(10)[2:8],
        torch.randn(10)[2:8:2],
        torch.randn(10, 10),
        torch.randn(10, 10).t(),
        torch.randn(10, 10, 2),
        torch.randn(10, 10, 2).transpose(0, 1),
        torch.randn(10, 10, 2, 2),
        torch.randn(10, 10, 2, 2, 1),
        torch.randn(10, 10, 2, 2, 1, 3),
        torch.randn(10, 10, 2, 2, 1, 3, 2),
        torch.randn(10, 10, 2, 2, 1, 3, 2, 2),
    ]


def gen_ip_port():
    """
    returns a random (IP, Port) pair [(str, str)] to avoid conflict of multi-test at same time.
    """
    t = int(time.time())
    x = time.time() - t
    ip0 = "127"
    ip1 = t % 256
    ip2 = int(x * 256 % 256)
    ip3 = int(x * 256 * 256 % 256)
    # Port 32768-60999 are used by Linux system
    # Port 0-10000 are used by local system or custom apps
    port = t // 256 % (32768 - 10000) + 10000
    return f"{ip0}.{ip1}.{ip2}.{ip3}", f"{port}"


def get_all_support_types():
    return [torch.float32, torch.int32, torch.int64]


def get_all_support_types_withfp16():
    return [torch.float16, torch.float32, torch.int32, torch.int64]


def get_all_types():
    return [
        torch.bool,
        torch.uint8,
        torch.float32,
        torch.int32,
        torch.float64,
        torch.int64,
    ]


class Comparator:
    """
    Base class used for comparing MUSA results and CPU golden results.
    """

    def __init__(self):
        self._comparator = None

    def __call__(self, result, golden):
        """Compare MUSA results and CPU results.
        Args:
            result: MUSA result.
            golden: CPU result.
        Returns:
            A bool value indicating whether computing result is right.
        """
        if self._comparator is None:
            raise NotImplementedError("Comparator is not implemented by a subclass")

        if not isinstance(self._comparator, (Callable, types.LambdaType)):
            raise TypeError("self._comparator must be a callable type")

        return self._comparator(result, golden) and result.shape == golden.shape


class DefaultComparator(Comparator):
    """The default comparator"""

    def __init__(self, abs_diff=1e-8, rel_diff=1e-5, equal_nan=False):
        """
        Use both relative and absolute tolerance to compare the result and golden.
        """
        super().__init__()
        self._comparator = lambda result, golden: torch.allclose(
            result, golden, rtol=rel_diff, atol=abs_diff, equal_nan=equal_nan
        )


class AbsDiffComparator(Comparator):
    """The absolute difference comparator."""

    def __init__(self, abs_diff=1e-3):
        """
        Use absolute tolerance to compare the result and golden.
        """
        super().__init__()
        self._comparator = (
            lambda result, golden: torch.abs(golden - result).max() < abs_diff
        )


class QuantizedComparator(Comparator):
    """
    The quantized comparator

    Use both relative and absolute tolerance to compare the dequantized value and
    quantized value of result and golden.
    """

    def __init__(
        self, abs_diff=1e-6, rel_diff=1e-5, equal_nan=False, is_per_tensor=True
    ):
        super().__init__()
        if is_per_tensor:
            self._comparator = lambda result, golden: (
                torch.allclose(
                    result.dequantize(),
                    golden.dequantize(),
                    rtol=rel_diff,
                    atol=abs_diff,
                    equal_nan=equal_nan,
                )
                and torch.allclose(
                    result.int_repr(),
                    golden.int_repr(),
                    rtol=rel_diff,
                    atol=abs_diff,
                    equal_nan=equal_nan,
                )
                and abs(result.q_scale() - golden.q_scale())
                <= abs_diff + rel_diff * golden.q_scale()
                and result.q_zero_point() == golden.q_zero_point()
            )
        else:
            self._comparator = lambda result, golden: (
                torch.allclose(
                    result.dequantize(),
                    golden.dequantize(),
                    rtol=rel_diff,
                    atol=abs_diff,
                    equal_nan=equal_nan,
                )
                and torch.allclose(
                    result.int_repr(),
                    golden.int_repr(),
                    rtol=rel_diff,
                    atol=abs_diff,
                    equal_nan=equal_nan,
                )
                and torch.allclose(
                    result.q_per_channel_scales(), golden.q_per_channel_scales()
                )
                and torch.allclose(
                    result.q_per_channel_zero_points(),
                    golden.q_per_channel_zero_points(),
                )
                and result.q_per_channel_axis() == golden.q_per_channel_axis()
            )


class RelDiffComparator(Comparator):
    """The relative difference comparator."""

    def __init__(self, rel_diff=1e-5):
        """
        Use relative tolerance to compare the result and golden.
        """
        super().__init__()
        self._comparator = (
            lambda result, golden: torch.abs((golden - result) / golden).max()
            < rel_diff
        )


class OpTest:
    """Infrastructure used for handling with op test.
    Args:
        func (function): Function used to invoke op.
        input_args (list): Input arguments for op.
        comparators (list): Comparator used to compare results.
        ignored_result_indices (list): Indices of results which will be ignored when comparing.
        seed (int): random seed may be used in nn.Module's initialization.
    """

    def __init__(
        self,
        func=None,
        input_args=None,
        comparators=DefaultComparator(equal_nan=True),
        ignored_result_indices=None,
        seed=42,
    ):
        assert func is not None, "no function defined."
        self._func = func
        self._input_args = input_args
        self._comparators = [comparators]
        self._ignored_result_indices = ignored_result_indices
        self._seed = seed

    def set_random_seed(self):
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

    def _call_func(
        self,
        inputs,
        device,
        train: bool = False,
        test_out: bool = False,
        fp16: bool = False,
    ):
        """Run op on specific device.
        Args:
            inputs (dict): Inputs arguments for op.
            device (str): Device where op will be ran.
            train (bool): Whether to test backward.
            test_out (bool): Whether to test op in out-of-place.
        Returns:
            Computing result in numpy format.
        """

        res = []
        grad = []
        mode_context = nullcontext() if train else torch.set_grad_enabled(False)
        input_args = {}
        with ExitStack() as stack:
            stack.enter_context(mode_context)
            for k in self._input_args:
                if isinstance(self._input_args[k], torch.Tensor):
                    input_args[k] = self._input_args[k].to(device).clone()
                    if fp16 and input_args[k].dtype == torch.float32:
                        input_args[k] = input_args[k].to(torch.float16)
                else:
                    input_args[k] = self._input_args[k]

                if (
                    train
                    and isinstance(input_args[k], torch.Tensor)
                    and input_args[k].requires_grad
                ):
                    input_args[k].retain_grad()
                    if input_args[k].grad is not None:
                        input_args[k].grad.zero_()

            if inputs is None:
                reduce = self._func(**input_args)
                if train:
                    if isinstance(reduce, (list, tuple)):
                        reduce = reduce[0]
                    reduce.sum().backward()
            elif isinstance(inputs, list):
                inputs_list = []
                for _, value in enumerate(inputs):
                    if isinstance(value, torch.Tensor):
                        inputs_list.append(value.to(device))
                    elif isinstance(value, np.ndarray):
                        tensor = torch.from_numpy(value).to(device)
                        inputs_list.append(tensor)
                    else:
                        inputs_list.append(value)
                reduce = self._func(*inputs_list)
            elif isinstance(inputs, dict):
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(device)
                    if isinstance(inputs[k], np.ndarray):
                        inputs[k] = torch.from_numpy(inputs[k]).to(device)
                    if train and inputs[k].requires_grad:
                        inputs[k].retain_grad()
                        if inputs[k].grad is not None:
                            inputs[k].grad.zero_()
                # For models with learnable parameters such as nn.Conv2d, it is necessary to
                # ensure that the model's parameters are consistent when performing calculations
                # on CPU and MUSA respectively, so we use same seed here.
                self.set_random_seed()
                func = self._func(**input_args)
                func.to(device)
                reduce = func(**inputs)
                if train:
                    if isinstance(reduce, (list, tuple)):
                        reduce = reduce[0]
                    reduce.sum().backward()
                    for _, val in inputs.items():
                        if val.requires_grad:
                            grad.append(val.grad.cpu())
            else:
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                if isinstance(inputs, np.ndarray):
                    inputs = torch.from_numpy(inputs).to(device)
                reduce = self._func(inputs)

            for k in input_args:
                if (
                    train
                    and isinstance(input_args[k], torch.Tensor)
                    and input_args[k].requires_grad
                ):
                    grad.append(input_args[k].grad.cpu())

            if isinstance(reduce, (tuple, list)):
                for val in reduce:
                    res.append(val.to("cpu"))
            else:
                res.append(reduce.to("cpu"))
            if test_out and "out" in input_args:
                res.append(input_args["out"].to("cpu"))
            for i in grad:
                res.append(i.clone())
            return res

    def compare_res(self, res1, res2):
        for i, (m_r, c_r) in enumerate(zip(res1, res2)):
            if self._ignored_result_indices and i in self._ignored_result_indices:
                continue
            if c_r.dtype == torch.float16:
                c_r = c_r.float()
            if m_r.dtype == torch.float16:
                m_r = m_r.float()
            for comparator in self._comparators:
                assert c_r.shape == m_r.shape
                assert c_r.dtype == m_r.dtype
                assert comparator(c_r, m_r)

    def check_musafp16_vs_musafp32(self, inputs=None, train=False, test_out=False):
        fp32_res = self._call_func(inputs, "musa", train, test_out)
        fp16_res = self._call_func(inputs, "musa", train, test_out, True)
        self.compare_res(fp32_res, fp16_res)

    def check_result(self, inputs=None, train=False, test_out=False):
        """Run op and compare computing results.
        Args:
            inputs (dict): Inputs arguments for op.
            train (bool): Whether to test backward.
            test_out (bool): Whether to test op in out-of-place.
        Returns:
            None.
        """
        cpu_res = self._call_func(inputs, "cpu", train, test_out)
        mtgpu_res = self._call_func(inputs, "musa", train, test_out)
        self.compare_res(cpu_res, mtgpu_res)
