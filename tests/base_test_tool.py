import numpy as np
import torch as pt
from contextlib import ExitStack, nullcontext
import torch_musa


class Comparator(object):
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
            raise NotImplemented
        else:
            return self._comparator(result, golden)


class DefaultComparator(Comparator):
    def __init__(self, abs_diff=1e-3, rel_diff=1e-5, equal_nan=False):
        """
        Use both relative and absolute tolerance to compare the result and golden.
        """
        super().__init__()
        self._comparator = lambda result, golden: np.allclose(
            result, golden, rtol=rel_diff, atol=abs_diff, equal_nan=equal_nan
        )


class AbsDiffComparator(Comparator):
    def __init__(self, abs_diff=1e-3):
        """
        Use absolute tolerance to compare the result and golden.
        """
        super().__init__()
        self._comparator = (
            lambda result, golden: np.abs(golden - result).max() < abs_diff
        )


class RelDiffComparator(Comparator):
    def __init__(self, rel_diff=1e-3):
        """
        Use relative tolerance to compare the result and golden.
        """
        super().__init__()
        self._comparator = (
            lambda result, golden: np.abs((golden - result) / golden).max() < rel_diff
        )

class OpTest(object):
    """Infrastructure used for handling with op test.

    Args:
        func (function): Function used to invoke op.
        input_args (list): Input arguments for op.
        comparators (list): Comparator used to compare results.
        ignored_result_indices (list): Indices of results which will be ignored when comparing.

    """
    def __init__(
        self, func=None, input_args=[], comparators=[DefaultComparator(equal_nan=True)], ignored_result_indices=[]
    ):
        assert func is not None, "no function defined."
        self._func = func
        self._input_args = input_args
        self._comparators = comparators
        self._ignored_result_indices = ignored_result_indices

    def _call_func(self, inputs, device, train: bool = False, test_out: bool = False):
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
        mode_context = (nullcontext() if train else pt.set_grad_enabled(False))
        with ExitStack() as stack:
            stack.enter_context(mode_context)
            for k in self._input_args:
                if isinstance(self._input_args[k], pt.Tensor):
                    self._input_args[k] = self._input_args[k].to(device)
                if isinstance(self._input_args[k], np.ndarray):
                    self._input_args[k] = pt.from_numpy(self._input_args[k]).to(device)
                elif isinstance(self._input_args[k], list):
                    for i in range(len(self._input_args[k])):
                        if isinstance(self._input_args[k][i], np.ndarray):
                            self._input_args[k][i] = pt.Tensor(
                                self._input_args[k][i]
                            ).to(device)
                            self._input_args[k][i].retain_grad()
                            if self._input_args[k][i].grad is not None:
                                self._input_args[k][i].grad.zero_()

                if train and isinstance(self._input_args[k], pt.Tensor) and \
                    self._input_args[k].requires_grad:
                    self._input_args[k].retain_grad()
                    if self._input_args[k].grad is not None:
                        self._input_args[k].grad.zero_()

            if inputs is None:
                r_ = self._func(**self._input_args)
                if train:
                    r_.sum().backward()
            elif isinstance(inputs, list):
                inputs_list = list()
                for index, value in enumerate(inputs):
                    if isinstance(value, pt.Tensor):
                        inputs_list.append(value.to(device))
                    if isinstance(value, np.ndarray):
                        tensor = pt.from_numpy(value).to(device)
                        inputs_list.append(tensor)
                r_ = self._func(*inputs_list)
            elif isinstance(inputs, dict):
                for k in inputs:
                    if isinstance(inputs[k], pt.Tensor):
                        inputs[k] = inputs[k].to(device)
                    if isinstance(inputs[k], np.ndarray):
                        inputs[k] = pt.from_numpy(inputs[k]).to(device)
                    if train and inputs[k].requires_grad:
                        inputs[k].retain_grad()
                        if type(inputs[k].grad) != type(None):
                            inputs[k].grad.zero_()
                f_ = self._func(**self._input_args)
                f_.to(device)
                r_ = f_(**inputs)
                if train:
                    r_.sum().backward()
                    for k, v in inputs.items():
                        if v.requires_grad:
                            grad.append(v.grad.cpu().detach().numpy())
            else:
                if isinstance(inputs, pt.Tensor):
                    inputs = inputs.to(device)
                if isinstance(inputs, np.ndarray):
                    inputs = pt.from_numpy(inputs).to(device)
                r_ = self._func(inputs)

            for k in self._input_args:
                if train and isinstance(self._input_args[k], pt.Tensor) and \
                    self._input_args[k].requires_grad:
                        grad.append(self._input_args[k].grad.cpu().detach().numpy())

            if isinstance(r_, tuple) or isinstance(r_, list):
                for r in r_:
                    res.append(r.to("cpu").detach().numpy())
            else:
                res.append(r_.to("cpu").detach().numpy())
            if test_out and "out" in self._input_args:
                res.append(self._input_args["out"].to("cpu").detach().numpy())
            for i in grad:
                res.append(i.copy())
            return res

    def __call__(self, inputs, train: bool = False, test_out: bool = False):
        """Run op and compare computing results.

        Args:
            inputs (dict): Inputs arguments for op.
            train (bool): Whether to test backward.
            test_out (bool): Whether to test op in out-of-place.

        Returns:
            None.

        """

        cpu_res = self._call_func(inputs, "cpu", train, test_out)
        mtgpu_res = self._call_func(inputs, "mtgpu", train, test_out)
        for i, (m_r, c_r) in enumerate(zip(mtgpu_res, cpu_res)):
            if i in self._ignored_result_indices:
                continue
            for comparator in self._comparators:
                assert c_r.shape == m_r.shape
                assert c_r.dtype == m_r.dtype
                assert comparator(c_r, m_r)
