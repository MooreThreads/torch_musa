"""Some basic functions for tests."""

# pylint: disable=too-few-public-methods, too-many-locals, too-many-statements, consider-using-dict-items
# pylint: disable=unused-import, too-many-branches, not-callable, missing-function-docstring
# pylint: disable=unidiomatic-typecheck, unused-variable

from contextlib import ExitStack, nullcontext
import random
import copy
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
        torch.randn(10, 0),
        torch.randn(10, 10).t(),
        torch.randn(10, 10, 2),
        torch.randn(10, 10, 2).transpose(0, 1),
        torch.randn(10, 10, 2, 2),
        torch.randn(10, 10, 2, 2, 1),
        torch.randn(10, 10, 2, 2, 1, 3),
        torch.randn(10, 10, 2, 2, 1, 3, 2),
        torch.randn(10, 10, 2, 2, 1, 3, 2, 2),
        torch.randn(10, 4, 2, 3).to(memory_format=torch.channels_last),
        torch.randn(10, 4, 1, 1).to(memory_format=torch.channels_last),
        torch.randn(10, 6, 1, 2).to(memory_format=torch.channels_last),
        torch.randn(10, 1, 3, 0).to(memory_format=torch.channels_last),
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


def get_float_types():
    return [torch.float32, torch.half, torch.bfloat16]


def get_all_types():
    return [
        torch.bool,
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.int32,
        torch.float64,
        torch.int64,
    ]


def _complex_musa_to_cpu_adjust(musa_complex: torch.Tensor) -> torch.Tensor:
    musa_real = torch.view_as_real(musa_complex)
    cpu_real = musa_real.to("cpu")
    cpu_complex = torch.view_as_complex(cpu_real)
    return cpu_complex


def _complex_cpu_to_musa_adjust(
    cpu_complex: torch.Tensor, musa_device: str
) -> torch.Tensor:
    cpu_real = torch.view_as_real(cpu_complex)
    musa_real = cpu_real.to(musa_device)
    musa_complex = torch.view_as_complex(musa_real)
    return musa_complex


class Comparator:
    """
    Base class used for comparing MUSA results and CPU golden results.
    """

    def __init__(self):
        self._comparator = None
        self._atol = 1e-5
        self._rtol = 1e-5
        self._equal_nan = False

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

        return (
            self._comparator(result.cpu(), golden.cpu())
            and result.shape == golden.shape
        )

    def get_tolerance(self):
        return self._atol, self._rtol, self._equal_nan


class DefaultComparator(Comparator):
    """The default comparator"""

    def __init__(self, abs_diff=1e-7, rel_diff=1e-5, equal_nan=False):
        """
        Use both relative and absolute tolerance to compare the result and golden.
        """
        super().__init__()
        self._atol = abs_diff
        self._rtol = rel_diff
        self._equal_nan = equal_nan
        self._comparator = lambda result, golden: torch.allclose(
            result, golden, rtol=rel_diff, atol=abs_diff, equal_nan=equal_nan
        )


class BooleanComparator(Comparator):
    """The boolean difference comparator."""

    def __init__(self):
        """
        Use element-wise equality to compare the result and golden.
        """
        super().__init__()
        self._comparator = lambda result, golden: torch.all(
            torch.eq(result, golden)
        ).item()


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
        refer_func (function, optional): Refer function used to invoke op.
            make sure the parameter names of func and refer_func be identical.
        input_args (list): Input arguments for op.
        comparators (list): Comparator used to compare results.
        ignored_result_indices (list): Indices of results which will be ignored when comparing.
        seed (int): random seed may be used in nn.Module's initialization.
    """

    def __init__(
        self,
        func=None,
        refer_func=None,
        input_args=None,
        comparators=DefaultComparator(equal_nan=True),
        ignored_result_indices=None,
        seed=42,
    ):
        assert func is not None, "no function defined."
        self._func = func
        self._input_args = input_args
        self._refer_func = refer_func
        self._comparators = [comparators]
        self._ignored_result_indices = ignored_result_indices
        self._seed = seed
        self._input_args_for_out_check = self.deep_copy_dict(input_args)

    def set_random_seed(self):
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

    def deep_copy_dict(self, args):
        res = {}
        for k in args:
            if isinstance(args[k], torch.Tensor):
                res[k] = args[k].clone()
            elif isinstance(args[k], dict):
                res[k] = self.deep_copy_dict(args[k])
            elif isinstance(args[k], torch._C.Generator):
                res[k] = args[k]
            else:
                res[k] = copy.deepcopy(args[k])
        return res

    def _call_func(
        self,
        inputs,
        device,
        train: bool = False,
        test_out: bool = False,
        fp16: bool = False,
        bf16: bool = False,
        refer: bool = False,
    ):
        """Run op on specific device.
        Args:
            inputs (dict): Inputs arguments for op.
            device (str): Device where op will be ran.
            train (bool): Whether to test backward.
            test_out (bool): Whether to test op in out-of-place.
            refer (bool): Whether it is in a reference scenario.
            fp16 (bool): Whether to set inputs to fp16 dtype
            bf16 (bool): Whether to set inputs to bf16 dtype
                Note: either fp16 or bf16 can be true.
        Returns:
            Computing result in numpy format.
        """
        cur_func = self._func
        if refer and self._refer_func is not None:
            cur_func = self._refer_func
        assert int(fp16) + int(bf16) <= 1, "either bf16 and fp16 can be True"

        res = []
        grad = []
        mode_context = nullcontext() if train else torch.set_grad_enabled(False)
        input_args = {}
        with ExitStack() as stack:
            stack.enter_context(mode_context)
            for k in self._input_args:
                if isinstance(self._input_args[k], torch.Tensor):
                    if not self._input_args[k].is_complex():
                        input_args[k] = self._input_args[k].to(device).clone()
                        if fp16 and input_args[k].dtype == torch.float32:
                            input_args[k] = input_args[k].to(torch.float16)
                        elif bf16 and input_args[k].dtype == torch.float32:
                            input_args[k] = input_args[k].to(torch.bfloat16)
                    else:
                        str_input_device = str(self._input_args[k].device)
                        if str_input_device.startswith("musa") and device.startswith(
                            "cpu"
                        ):
                            input_args[k] = _complex_musa_to_cpu_adjust(
                                self._input_args[k]
                            )
                        elif str_input_device.startswith("cpu") and device.startswith(
                            "musa"
                        ):
                            input_args[k] = _complex_cpu_to_musa_adjust(
                                self._input_args[k], device
                            )
                        else:
                            input_args[k] = self._input_args[k]
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
                prev_addr = self.get_addr_list_of_args(input_args)
                reduce = cur_func(**input_args)
                if train:
                    if isinstance(reduce, (list, tuple)):
                        reduce = reduce[0]
                    reduce.sum().backward()
                post_addr = self.get_addr_list_of_args(input_args)
                assert (
                    prev_addr == post_addr
                ), "The position of tensor should not be changed"
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
                prev_addr = self.get_addr_list_of_args(inputs_list)
                reduce = cur_func(*inputs_list)
                post_addr = self.get_addr_list_of_args(inputs_list)
                assert (
                    prev_addr == post_addr
                ), "The position of tensor should not be changed"
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
                prev_addr = self.get_addr_list_of_args(input_args)
                func = cur_func(**input_args)
                post_addr = self.get_addr_list_of_args(input_args)
                assert (
                    prev_addr == post_addr
                ), "The position of tensor should not be changed"
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
                prev_addr = inputs.data_ptr()
                reduce = cur_func(inputs)
                post_addr = inputs.data_ptr()
                assert (
                    prev_addr == post_addr
                ), "The position of tensor should not be changed"
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
            elif isinstance(reduce, bool):
                res.append(reduce)
            else:
                if (
                    isinstance(reduce, torch.Tensor)
                    and reduce.is_complex()
                    and str(reduce.device).startswith("musa")
                ):
                    res.append(_complex_musa_to_cpu_adjust(reduce))
                else:
                    res.append(reduce.to("cpu"))
            if test_out and "out" in input_args:
                res.append(input_args["out"].to("cpu"))
            for i in grad:
                res.append(i.clone())
            return res

    def compare_res(self, res1, res2):
        for i, (m_r, c_r) in enumerate(zip(res1, res2)):
            if isinstance(m_r, bool):
                assert m_r == c_r
                return
            if self._ignored_result_indices and i in self._ignored_result_indices:
                continue
            if c_r.dtype == torch.float16:
                c_r = c_r.float()
            if c_r.dtype == torch.bfloat16:
                c_r = c_r.float()
            if m_r.dtype == torch.float16:
                m_r = m_r.float()
            if m_r.dtype == torch.bfloat16:
                m_r = m_r.float()
            for comparator in self._comparators:
                assert c_r.shape == m_r.shape
                assert c_r.dtype == m_r.dtype
                res = comparator(m_r, c_r)
                info_str = ""
                if not res:
                    atol, rtol, equal_nan = comparator.get_tolerance()
                    mask_t = ~torch.isclose(m_r, c_r, rtol, atol, equal_nan)
                    selected = torch.abs(c_r[mask_t] - m_r[mask_t])
                    info_str = (
                        f"Max abs error: {selected.max().item()} "
                        f"found in {i + 1}-th comparasion"
                    )

                assert res, info_str

    def check_musafp16_vs_musafp16(self, inputs=None, train=False, test_out=False):
        """
        Compare results of ref_func and func.
        This are designed for some ops could be validated by compositing small ops like SDP.
        """
        fp16_ref_res = self._call_func(
            inputs, "musa", train, test_out, refer=True, fp16=True
        )
        fp16_comp_res = self._call_func(inputs, "musa", train, test_out, fp16=True)
        self.compare_res(fp16_ref_res, fp16_comp_res)

    def check_musafp16_vs_musafp32(self, inputs=None, train=False, test_out=False):
        fp32_res = self._call_func(inputs, "musa", train, test_out, refer=True)
        fp16_res = self._call_func(inputs, "musa", train, test_out, True)
        self.compare_res(fp32_res, fp16_res)

    def check_musabf16_vs_musafp16(self, inputs=None, train=False, test_out=False):
        fp16_res = self._call_func(
            inputs, "musa", train, test_out, refer=True, fp16=True
        )
        bf16_res = self._call_func(inputs, "musa", train, test_out, bf16=True)
        self.compare_res(fp16_res, bf16_res)

    def check_result(self, inputs=None, train=False, test_out=False, **kwargs):
        """Run op and compare computing results.
        Args:
            inputs (dict): Inputs arguments for op.
            train (bool): Whether to test backward.
            test_out (bool): Whether to test op in out-of-place.
            kwargs: other arguments may be needed.
        """
        cpu_res = self._call_func(inputs, "cpu", train, test_out, refer=True, **kwargs)
        mtgpu_res = self._call_func(inputs, "musa", train, test_out, **kwargs)
        self.compare_res(cpu_res, mtgpu_res)

    def get_addr_list_of_args(self, args):
        addr_list = []
        if isinstance(args, dict):
            for k in args:
                # When the size of the out tensor does not meet expectations,
                # its address can be changed
                if k == "out":
                    continue
                if isinstance(args[k], torch.Tensor):
                    addr_list.append(args[k].data_ptr())
        elif isinstance(args, list):
            for index, element in enumerate(args):
                if isinstance(args[index], torch.Tensor):
                    addr_list.append(args[index].data_ptr())
        return addr_list

    def args_to_device(self, args, device, requires_grad=False):
        for k in args:
            if isinstance(args[k], torch.Tensor):
                args[k] = args[k].to(device)

                # only Tensors of floating point dtype can require gradients
                if requires_grad and args[k].dtype in (
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                ):
                    args[k] = args[k].requires_grad_()
        return args

    def args_to_dtype(self, args, dtype):
        for k in args:
            if isinstance(args[k], torch.Tensor):
                args[k] = args[k].to(dtype)
        return args

    def dim_0_tensor_compare(self, res1, res2):
        if res1.dtype == torch.float16:
            res1 = res1.float()
        if res1.dtype == torch.bfloat16:
            res1 = res1.float()
        if res2.dtype == torch.float16:
            res2 = res2.float()
        if res2.dtype == torch.bfloat16:
            res2 = res2.float()
        for comparator in self._comparators:
            assert res1.shape == res2.shape
            assert res1.dtype == res2.dtype
            res = comparator(res1, res2)
            info_str = ""
            if not res:
                atol, rtol, equal_nan = comparator.get_tolerance()
                mask_t = ~torch.isclose(res1, res2, rtol, atol, equal_nan)
                selected = torch.abs(res1[mask_t] - res2[mask_t])
                info_str = f"Max abs error: {selected.max().item()}"

            assert res, info_str

    def check_out_ops(self, bf16=False, fp16=False):
        cur_func = self._func

        # If there is no __ name__ attribute, it is torch.nn layer,
        # if the latest char is '_', it is inplace op
        if not hasattr(cur_func, "__name__") or cur_func.__name__[-1] == "_":
            return

        # Get output of normal ops
        musa_normal_input_args = self.args_to_device(
            copy.deepcopy(self._input_args_for_out_check), "musa"
        )
        if bf16:
            musa_normal_input_args = self.args_to_dtype(
                musa_normal_input_args, torch.bfloat16
            )
        elif fp16:
            musa_normal_input_args = self.args_to_dtype(
                musa_normal_input_args, torch.float16
            )
        normal_output = cur_func(**musa_normal_input_args)
        if isinstance(normal_output, torch.Tensor):

            # The address of out tensor that meets the expected size should not be changed
            musa_out_input_args = self.args_to_device(
                copy.deepcopy(self._input_args_for_out_check), "musa"
            )
            if bf16:
                musa_out_input_args = self.args_to_dtype(
                    musa_out_input_args, torch.bfloat16
                )
            elif fp16:
                musa_out_input_args = self.args_to_dtype(
                    musa_out_input_args, torch.float16
                )
            out_tensor = torch.empty_like(normal_output)
            prev_out_tensor_addr = out_tensor.data_ptr()
            musa_out_input_args["out"] = out_tensor
            out_output = cur_func(**musa_out_input_args)
            assert (
                prev_out_tensor_addr == musa_out_input_args["out"].data_ptr()
            ), "The position \
                of tensor should not be changed"
            assert (
                prev_out_tensor_addr == out_output.data_ptr()
            ), "The position of tensor should \
                not be changed"
            if out_output.dim() == 0:
                self.dim_0_tensor_compare(
                    out_output.cpu(), musa_out_input_args["out"].cpu()
                )
                self.dim_0_tensor_compare(
                    normal_output.cpu(), musa_out_input_args["out"].cpu()
                )
            else:
                self.compare_res(out_output.cpu(), musa_out_input_args["out"].cpu())
                self.compare_res(normal_output.cpu(), musa_out_input_args["out"].cpu())

    def check_grad_fn(self, bf16=False, fp16=False):
        cur_func = self._func

        # a leaf Variable that requires grad can not be used in an in-place operation.
        if hasattr(cur_func, "__name__") and cur_func.__name__[-1] == "_":
            return

        # functions with out=... arguments don't support automatic differentiation.
        if "out" in self._input_args_for_out_check:
            return

        musa_normal_input_args = self.args_to_device(
            copy.deepcopy(self._input_args_for_out_check), "musa", requires_grad=True
        )
        if bf16:
            musa_normal_input_args = self.args_to_dtype(
                musa_normal_input_args, torch.bfloat16
            )
        elif fp16:
            musa_normal_input_args = self.args_to_dtype(
                musa_normal_input_args, torch.float16
            )

        cpu_normal_input_args = self.args_to_device(
            copy.deepcopy(self._input_args_for_out_check), "cpu", requires_grad=True
        )

        # Some CPU operators do not support fp16/bf16. When confirming grad_fn,
        # we do not pay attention to numerical precision, so we directly use fp32 on the CPU.
        for k in cpu_normal_input_args:
            if isinstance(
                cpu_normal_input_args[k], torch.Tensor
            ) and cpu_normal_input_args[k].dtype in (torch.float16, torch.bfloat16):
                cpu_normal_input_args[k] = cpu_normal_input_args[k].to(torch.float32)
        musa_output = cur_func(**musa_normal_input_args)
        cpu_output = cur_func(**cpu_normal_input_args)
        if hasattr(musa_output, "grad_fn"):
            assert id(musa_output.grad_fn.__class__) == id(
                cpu_output.grad_fn.__class__
            ), f"grad_fn of musa_output({musa_output.grad_fn}) is not same as \
                    cpu_output({cpu_output.grad_fn})!"


class InplaceOpChek:
    """Check if the position of the self tensor changes before and after
    the execution of the inplace operator, and the inplace ops res.
    This class does not manage tensor data types
    Args:
        func_name (string): The name of self tensor member function that will be executed
        self_tensor (tensor): Self tensor to execute the inplace op.
        input_args (dict): Input arguments for op.
        comparators (list): Comparator used to compare results.
    """

    def __init__(
        self,
        func_name: str,
        self_tensor,
        input_args: dict = None,
        comparators=None,
    ):
        self._func_name = func_name
        if not isinstance(self_tensor, torch.Tensor):
            self_tensor = torch.tensor(self_tensor)
        self._musa_self_tensor = self_tensor.clone().musa()
        self._cpu_self_tensor = self_tensor.clone().cpu()
        if input_args is not None:
            self._musa_input_args = self.args_to_device(
                copy.deepcopy(input_args), "musa"
            )
            self._cpu_input_args = self.args_to_device(copy.deepcopy(input_args), "cpu")
        else:
            self._musa_input_args = {}
            self._cpu_input_args = {}
        if comparators is None:
            self._comparators = [DefaultComparator(equal_nan=True)]
        else:
            self._comparators = comparators

    def check_address(self):
        """
        The purpose of this function is to confirm whether the position of
        the self tensor and parameters has changed before and after calling
        the inplace operator
        """
        musa_self_tensor = self._musa_self_tensor.clone()
        inplace_func = getattr(musa_self_tensor, self._func_name)

        # Obtain the position of self tensor and the tensor in the
        # parameters before calling the inplace operator
        origin_addr = musa_self_tensor.data_ptr()
        origin_input_args_addr = []
        musa_input_args = copy.deepcopy(self._musa_input_args)
        for k in musa_input_args:
            if isinstance(musa_input_args[k], torch.Tensor):
                origin_input_args_addr.append(musa_input_args[k].data_ptr())
        inplace_func(**musa_input_args)

        # Obtain the position of self tensor and the tensor
        # in the parameters after calling the inplace operator
        res_addr = musa_self_tensor.data_ptr()
        res_input_args_addr = []
        for k in musa_input_args:
            if isinstance(musa_input_args[k], torch.Tensor):
                res_input_args_addr.append(musa_input_args[k].data_ptr())

        assert origin_addr == res_addr, "The position of tensor should not be changed"
        assert (
            origin_input_args_addr == res_input_args_addr
        ), "The position of tensor \
            should not be changed"

    def args_to_device(self, args, device):
        for k in args:
            if isinstance(args[k], torch.Tensor):
                args[k] = args[k].to(device)
        return args

    def compare_res(self, m_r, c_r):
        if isinstance(m_r, bool):
            assert m_r == c_r
            return
        if c_r.dtype in (torch.float16, torch.bfloat16):
            c_r = c_r.float()
        if m_r.dtype in (torch.float16, torch.bfloat16):
            m_r = m_r.float()
        for comparator in self._comparators:
            assert c_r.shape == m_r.shape
            assert c_r.dtype == m_r.dtype
            res = comparator(m_r, c_r)
            info_str = ""
            if not res:
                atol, rtol, equal_nan = comparator.get_tolerance()
                mask_t = ~torch.isclose(m_r, c_r, rtol, atol, equal_nan)
                selected = torch.abs(c_r[mask_t] - m_r[mask_t])
                info_str = f"Max abs error: {selected.max().item()}"

            assert res, info_str

    def compare_dict(self, dict1, dict2):
        assert set(dict1.keys()) == set(dict2.keys())
        for k in dict1:
            assert type(dict1[k]) == type(dict2[k])
            if isinstance(dict1[k], torch.Tensor):
                self.compare_res(dict1[k].cpu(), dict2[k].cpu())
            elif isinstance(dict1[k], dict):
                self.compare_dict(dict1[k], dict2[k])
            elif isinstance(dict1[k], list):
                self.compare_list(dict1[k], dict2[k])
            else:
                assert dict1[k] == dict2[k]

    def compare_list(self, list1, list2):
        assert len(list1) == len(list2)
        for index, element in enumerate(list1):
            assert type(list1[index]) == type(list2[index])
            if isinstance(list1[index], list):
                self.compare_list(list1[index], list2[index])
            elif isinstance(list1[index], dict):
                self.compare_dict(list1[index], list2[index])
            elif isinstance(list1[index], torch.Tensor):
                self.compare_res(list1[index].cpu(), list2[index].cpu())
            else:
                assert list1[index] == list2[index]

    def args_to_float32(self, args):
        for k in args:
            if isinstance(args[k], torch.Tensor) and args[k].dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                args[k] = args[k].float()
        return args

    def check_res(self, cpu_to_fp32=False):
        """
        This function is used to confirm the correctness of the calculation
        results of the inplace operator.
        Args:
            cpu_to_fp32 (bool): Set true if the ops does not support bf16/fp16 in cpu.
        """
        musa_self_tensor = self._musa_self_tensor.clone()
        musa_input_args = copy.deepcopy(self._musa_input_args)
        inplace_func = getattr(musa_self_tensor, self._func_name)
        inplace_func(**musa_input_args)

        cpu_self_tensor = self._cpu_self_tensor.clone()
        cpu_input_args = copy.deepcopy(self._cpu_input_args)

        # we change dtype to float32 because some ops are not
        # supported for torch.float16/torch.bfloat16 in cpu
        if cpu_to_fp32:
            if cpu_self_tensor.dtype in (torch.float16, torch.bfloat16):
                cpu_self_tensor = cpu_self_tensor.float()
            cpu_input_args = self.args_to_float32(cpu_input_args)

        inplace_func = getattr(cpu_self_tensor, self._func_name)
        inplace_func(**cpu_input_args)

        self.compare_res(musa_self_tensor.cpu(), cpu_self_tensor)
        self.compare_dict(musa_input_args, cpu_input_args)
