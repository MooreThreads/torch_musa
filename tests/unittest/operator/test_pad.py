"""Test padding operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from typing import (
    List,
    Tuple,
    Union,
    Callable,
)
import torch
from torch import nn
import pytest

from torch_musa import testing


support_dtypes = testing.get_float_types() + [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
]


pad1d_configs = [
    # shape, (padding_left, padding_right)
    [(16, 16), (2, 1)],
    [(4, 16, 16), (1, 2)],
]

pad2d_configs = [
    # shape, (padding_left, padding_right, padding_top, padding_bottom)
    [(4, 16, 16), (1, 2, 0, 1)],
    [(2, 4, 64, 64), (2, 1, 1, 0)],
]

pad3d_configs = [
    # shape, (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    [(2, 4, 64, 64), (2, 1, 1, 0, 1, 2)],
    [(2, 2, 4, 64, 64), (2, 1, 1, 0, 1, 2)],
]

PADDING_MOD = ["constant", "circular", "replication", "reflection"]

pad1d_mapping = {
    "constant": nn.ConstantPad1d,
    "circular": nn.CircularPad1d,
    "replication": nn.ReplicationPad1d,
    "reflection": nn.ReflectionPad1d,
}

pad2d_mapping = {
    "constant": nn.ConstantPad2d,
    "circular": nn.CircularPad2d,
    "replication": nn.ReplicationPad2d,
    "reflection": nn.ReflectionPad2d,
}

pad3d_mapping = {
    "constant": nn.ConstantPad3d,
    "circular": nn.CircularPad3d,
    "replication": nn.ReplicationPad3d,
    "reflection": nn.ReflectionPad3d,
}


class TestPad:
    """Test suit of torch.nn.functional.pad"""

    def _run_pad(self, func: Callable, input_args: dict, train: bool = False):
        op_test = testing.OpTest(func=func, input_args=input_args)

        if input_args["input"].dtype == torch.float16:
            op_test.check_musafp16_vs_cpufp32(train=train)
        else:
            op_test.check_result(train=train)

    @pytest.mark.parametrize("config", pad1d_configs)
    @pytest.mark.parametrize("mode", PADDING_MOD)
    @pytest.mark.parametrize("dtype", support_dtypes)
    @pytest.mark.parametrize("train", [True, False])
    def test_pad1d(
        self,
        config: Union[List[int], Tuple[int, ...]],
        mode: str,
        dtype: torch.dtype,
        train: bool,
    ) -> None:
        if dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            train = False
        shape, pad = config
        pad_module = pad1d_mapping[mode]
        if mode == "constant":
            pad_inst = pad_module(pad, 1.0)
        else:
            pad_inst = pad_module(pad)
        input_tensor = torch.randn(shape).to(dtype).requires_grad_(train)
        input_args = {"input": input_tensor}
        self._run_pad(pad_inst, input_args, train)

    @pytest.mark.parametrize("config", pad2d_configs)
    @pytest.mark.parametrize("mode", PADDING_MOD)
    @pytest.mark.parametrize("dtype", support_dtypes)
    @pytest.mark.parametrize("train", [True, False])
    @pytest.mark.parametrize("is_channels_last", [True, False])
    def test_pad2d(
        self,
        config: Union[List[int], Tuple[int, ...]],
        mode: str,
        dtype: torch.dtype,
        train: bool,
        is_channels_last: bool,
    ) -> None:
        if dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            train = False
        shape, pad = config
        pad_module = pad2d_mapping[mode]
        if mode == "constant":
            pad_inst = pad_module(pad, 1.0)
        else:
            pad_inst = pad_module(pad)
        if is_channels_last and len(shape) == 4:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        input_tensor = (
            torch.randn(shape)
            .to(dtype)
            .to(memory_format=memory_format)
            .requires_grad_(train)
        )
        input_args = {"input": input_tensor}
        self._run_pad(pad_inst, input_args, train)

    @pytest.mark.parametrize("config", pad3d_configs)
    @pytest.mark.parametrize("mode", PADDING_MOD)
    @pytest.mark.parametrize("dtype", support_dtypes)
    @pytest.mark.parametrize("train", [True, False])
    @pytest.mark.parametrize("is_channels_last", [True, False])
    def test_pad3d(
        self,
        config: Union[List[int], Tuple[int, ...]],
        mode: str,
        dtype: torch.dtype,
        train: bool,
        is_channels_last: bool,
    ) -> None:
        if dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            train = False
        shape, pad = config
        pad_module = pad3d_mapping[mode]
        if mode == "constant":
            pad_inst = pad_module(pad, 1.0)
        else:
            pad_inst = pad_module(pad)
        if is_channels_last:
            memory_format = (
                torch.channels_last if len(shape) == 4 else torch.channels_last_3d
            )
        else:
            memory_format = torch.contiguous_format
        input_tensor = (
            torch.randn(shape)
            .to(dtype)
            .to(memory_format=memory_format)
            .requires_grad_(train)
        )
        input_args = {"input": input_tensor}
        self._run_pad(pad_inst, input_args, train)


pad2d_configs = [
    # shape: (N, C, H, W), padding: (left, right, top, bottom)
    [(4, 16, 16, 16), (0, 0, 0, 0)],
    [(4, 16, 16, 16), (1, 2, 0, 1)],
    [(2, 4, 64, 64), (0, 0, 0, 0)],
    [(2, 4, 64, 64), (2, 1, 1, 0)],
]

support_dtypes = [torch.float32]


@pytest.mark.parametrize("config", pad2d_configs)
@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("is_channels_last", [True, False])
def test_replication_pad2d_backward_grad_input(config, dtype, train, is_channels_last):
    shape, pad = config
    if len(shape) == 4 and is_channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    x = torch.randn(shape, dtype=dtype, requires_grad=train).to(
        memory_format=memory_format
    )

    pad_module = nn.ReplicationPad2d(pad)
    y = pad_module(x)

    grad_output = torch.randn_like(y)
    grad_input = torch.zeros_like(x)

    grad_input1 = torch.ops.aten.replication_pad2d_backward(grad_output, x, pad)
    with torch.no_grad():
        grad_input = torch.ops.aten.replication_pad2d_backward.grad_input(
            grad_output, x, pad, grad_input=grad_input
        )

    torch.testing.assert_close(grad_input, grad_input1, rtol=1e-6, atol=1e-6)
