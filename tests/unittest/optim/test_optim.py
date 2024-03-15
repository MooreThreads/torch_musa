"""Test optimizers, some of these code was borrowed from pytorch official unit test"""
# pylint: disable=missing-class-docstring, missing-function-docstring
import copy
import numbers
from typing import (
    Optional,
    Tuple,
    Union,
)
import torch
from torch import optim
import torch_musa


# Some analysis of tolerance by logging tests from test_torch.py can be found in
# https://github.com/pytorch/pytorch/pull/32538.
# {dtype: (rtol, atol)}
_DTYPE_PRECISIONS = {
    torch.float32: (1.3e-6, 1e-5),
}

def default_tolerances(*inputs: Union[torch.Tensor, torch.dtype]) -> Tuple[float, float]:
    dtypes = []
    for _input in inputs:
        if isinstance(_input, torch.Tensor):
            dtypes.append(_input.dtype)
        elif isinstance(_input, torch.dtype):
            dtypes.append(_input)
        else:
            raise TypeError(
                f"Expected a torch.Tensor or a torch.dtype, but got {type(_input)} instead."
            )
    dtype_precisions = _DTYPE_PRECISIONS
    rtols, atols = zip(*[dtype_precisions.get(dtype, (0.0, 0.0)) for dtype in dtypes])
    return max(rtols), max(atols)


def get_tolerances(
    *inputs: Union[torch.Tensor, torch.dtype],
    rtol: Optional[float],
    atol: Optional[float]
) -> Tuple[float, float]:
    if (rtol is None) ^ (atol is None):
        raise ValueError("Both 'rtol' and 'atol' must be either specified or omitted")
    if rtol is not None and atol is not None:
        return rtol, atol
    return default_tolerances(*inputs)


class TestOptim:
    def __init__(self, atol=None, rtol=None):
        self.atol = atol
        self.rtol = rtol

    def assert_equal(self, golen, result, equal_nan=False):
        rtol, atol = get_tolerances(*[result, golen], rtol=self.rtol, atol=self.atol)
        matches = torch.isclose(result, golen, rtol=rtol, atol=atol, equal_nan=equal_nan)
        assert torch.all(matches)

    def _test_multi_tensor_optimizers(self, configs):
        if not torch_musa.is_available():
            return
        k_iterations = 7
        device = "musa"
        for optim_ctr, params in configs:
            res, state = [], []
            for foreach in (False, True):
                input_t = torch.tensor(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32, device=device
                ).reshape(3, 2)
                torch.manual_seed(1)
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(3, 1),
                    torch.nn.Sigmoid(),
                )
                model.to(dtype=torch.float32, device=device)
                params_with_flag = copy.deepcopy(params)
                params_with_flag["foreach"] = foreach

                optimizer = optim_ctr(
                    model.parameters(), **params_with_flag
                )
                for _ in range(k_iterations):
                    optimizer.zero_grad()
                    output = model(input_t)
                    loss = output.sum()
                    loss.backward()
                    optimizer.step()

                state.append(optimizer.state)
                res.append(model.parameters())

            st_state, mt_state = state[0], state[1]
            for st_p, mt_p in zip(res[0], res[1]):
                self.assert_equal(st_p.cpu(), mt_p.cpu())

                st_p_state = st_state[st_p]
                mt_p_state = mt_state[mt_p]
                for k in st_p_state:
                    golden = st_p_state[k]
                    actual = mt_p_state[k]
                    if k == "step" and isinstance(actual, numbers.Number):
                        actual = torch.Tensor([actual])
                        golden = torch.Tensor([golden])
                    assert isinstance(golden, torch.Tensor) and isinstance(actual, torch.Tensor)
                    self.assert_equal(golden.cpu(), actual.cpu())
                    # assert torch.allclose(golden.cpu(), actual.cpu())

    def test_multi_tensor_optimizers(self):
        configs = [
            (optim.Adam, {"weight_decay": 1.0, "amsgrad": False, "fused": False}),
            (optim.Adam, {"weight_decay": 1.0, "amsgrad": True, "fused": False}),
            (optim.Adam, {"weight_decay": 0.0, "amsgrad": False, "fused": False}),
            (optim.Adam, {"weight_decay": 0.0, "amsgrad": True, "fused": False}),
            (optim.AdamW, {"weight_decay": 1.0, "amsgrad": False}),
            (optim.AdamW, {"weight_decay": 1.0, "amsgrad": True}),
            (optim.AdamW, {"weight_decay": 0.0, "amsgrad": False}),
            (optim.AdamW, {"weight_decay": 0.0, "amsgrad": True}),
            (optim.NAdam, {"weight_decay": 0.0, "momentum_decay": 6e-3}),
            (optim.NAdam, {"weight_decay": 1.0, "momentum_decay": 6e-3}),
            (optim.NAdam, {"weight_decay": 0.0, "momentum_decay": 4e-3}),
            (optim.NAdam, {"weight_decay": 0.01, "momentum_decay": 4e-3}),
            (
                optim.SGD,
                {"lr": 0.2, "momentum": 1, "dampening": 0.5, "weight_decay": 1, "nesterov": False}
            ),
            (
                optim.SGD,
                {"lr": 0.2, "momentum": 1, "dampening": 0, "weight_decay": 1, "nesterov": True}
            ),
            (optim.RAdam, {"weight_decay": 0., "eps": 1e-6}),
            (optim.RAdam, {"weight_decay": 0.}),
            (optim.RAdam, {"weight_decay": 1., "eps": 1e-6}),
            (optim.RAdam, {"weight_decay": 1.}),
            (optim.RMSprop, {"weight_decay": 1, "momentum": 1, "centered": True}),
            (optim.RMSprop, {"weight_decay": 1, "momentum": 0, "centered": True}),
            (optim.RMSprop, {"weight_decay": 1, "momentum": 1, "centered": False}),
            (optim.RMSprop, {"weight_decay": 0, "momentum": 1, "centered": False}),
            (optim.Rprop, {"lr": 1e-2, "etas": (0.5, 1.2), "step_sizes": (1e-6, 50)}),
            (optim.ASGD, {"weight_decay": 0}),
            (optim.ASGD, {"weight_decay": 1}),
            (optim.Adamax, {"weight_decay": 0}),
            (optim.Adamax, {"weight_decay": 1}),
            (optim.Adadelta, {"weight_decay": 0}),
            (optim.Adadelta, {"weight_decay": 1}),
            (optim.Adagrad, {"weight_decay": 0}),
            (optim.Adagrad, {"weight_decay": 1}),
        ]
        self._test_multi_tensor_optimizers(configs)


def test_multi_tensor_optimizer():
    TestOptim().test_multi_tensor_optimizers()
