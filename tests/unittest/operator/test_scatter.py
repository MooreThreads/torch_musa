"""Test scatter operators."""

# pylint: disable=missing-function-docstring,missing-module-docstring,unused-import,redefined-builtin,redefined-outer-name
import copy
import torch
import pytest
from torch_musa import testing
import torch_musa

inputs = [
    {
        "input": torch.randn(12),
        "dim": 0,
        "index": torch.tensor([0, 1, 2, 0]),
        "src": torch.arange(1, 10),
    },
    {
        "input": torch.randn(5, 5),
        "dim": 0,
        "index": torch.tensor([[0, 1, 2, 0]]),
        "src": torch.arange(1, 11).reshape((2, 5)),
    },
    {
        "input": torch.randn(5, 5),
        "dim": 1,
        "index": torch.tensor([[0, 1, 2], [0, 1, 1]]),
        "src": torch.arange(1, 11).reshape((2, 5)),
    },
]

dtypes = [
    torch.float32,
    torch.float16,
    torch.float64,
    torch.int64,
    torch.int8,
    torch.int32,
    torch.short,
    torch.bool,
    torch.uint8,
]

reduce = ["add", "multiply"]

if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize("dtype", dtypes)
def test_scatter_(inputs, dtype):
    inputs["input"] = inputs["input"].to(dtype)
    inputs["src"] = inputs["src"].to(dtype)
    test = testing.OpTest(
        func=torch.scatter,
        input_args=inputs,
    )
    test.check_result()
    test.check_out_ops()
    inplace_input = copy.deepcopy(inputs)
    self_tensor = inplace_input["input"]
    inplace_input.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.scatter.__name__ + "_",
        self_tensor=self_tensor,
        input_args=inplace_input,
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize("dtype", dtypes)
def test_scatter_value(inputs, dtype):
    input = inputs["input"].clone().to(dtype)
    dim = inputs["dim"]
    index = inputs["index"].clone()
    musa_input = input.musa()
    cpu_res = input.scatter_(dim, index, 1.0)
    musa_res = musa_input.scatter_(dim, index.musa(), 1.0)
    assert torch.dist(cpu_res.float(), musa_res.cpu().float()) < 1e-3


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize("dtype", dtypes)
def test_scatter_add(inputs, dtype):
    inputs["input"] = inputs["input"].to(dtype)
    inputs["src"] = inputs["src"].to(dtype)
    test = testing.OpTest(
        func=torch.scatter_add,
        input_args=inputs,
    )
    test.check_result()
    test.check_out_ops()
    inplace_input = copy.deepcopy(inputs)
    self_tensor = inplace_input["input"]
    inplace_input.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.scatter_add.__name__ + "_",
        self_tensor=self_tensor,
        input_args=inplace_input,
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config",
    [
        # self_shape, index_shape, src_shape, dim
        [(64,), (64,), (64,), 0],
        [(64,), (128,), (128,), 0],
        [(16, 64), (16, 64), (16, 64), 0],
        [(16, 64), (16, 32), (16, 32), 0],
        [(16, 64), (16, 32), (16, 128), 0],
    ],
)
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("include_self", [False, True])
def test_scatter_reduce(config, reduce, dtype, include_self):
    # not implemented for torch.bool
    if dtype == torch.bool:
        return
    self_shape, index_shape, src_shape, dim = config
    input_args = {
        "input": torch.randn(self_shape).to(dtype),
        "src": torch.randn(src_shape).to(dtype),
        "index": torch.randint(0, self_shape[dim], index_shape),
        "reduce": reduce,
        "include_self": include_self,
        "dim": dim,
    }
    if dtype == torch.float16:
        abs_diff, rel_diff = 5e-3, 5e-3
    elif dtype == torch.bfloat16:
        abs_diff, rel_diff = 5e-2, 5e-2
    else:
        abs_diff, rel_diff = 1e-5, 1e-6
    test = testing.OpTest(
        func=torch.scatter_reduce,
        input_args=input_args,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("reduce", reduce)
def test_scatter_value_reduce(inputs, dtype, reduce):
    if dtype == torch.bool:
        return
    input = inputs["input"].clone().to(dtype)
    dim = inputs["dim"]
    index = inputs["index"].clone()
    musa_input = input.musa()
    cpu_res = input.scatter_(dim, index, 1.0, reduce=reduce)
    musa_res = musa_input.scatter_(dim, index.musa(), 1.0, reduce=reduce)
    assert torch.dist(cpu_res.float(), musa_res.cpu().float()) < 1e-3
