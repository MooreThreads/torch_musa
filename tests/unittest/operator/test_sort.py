"""Test sort operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    [torch.tensor(1.5), 0],
    [torch.randn(0), 0],
    [torch.randn(10), 0],
    [torch.randn(10)[2:8], -1],
    [torch.randn(10, 10), 1],
    [torch.randn(10, 0), 0],
    [torch.randn(10, 10).t(), 1],
    [torch.randn(10, 10, 2, 6, 1, 3), 3],
    [torch.randn(10, 4, 1, 1).to(memory_format=torch.channels_last), 0],
]

dtypes = [
    torch.float16,
    torch.float32,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
]

if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("stable", [True])
def test_sort(input_data, dtype, descending, stable):
    input_args = {}
    input_args["input"] = input_data[0].to(dtype)
    dim = input_data[1]
    input_args["dim"] = dim
    input_args["descending"] = descending
    input_args["stable"] = stable
    test = testing.OpTest(func=torch.sort, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
    else:
        test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("stable", [True])
def test_argsort(input_data, dtype, descending, stable):
    input_args = {}
    input_args["input"] = input_data[0].to(dtype)
    dim = input_data[1]
    input_args["dim"] = dim
    input_args["descending"] = descending
    input_args["stable"] = stable
    test = testing.OpTest(func=torch.argsort, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
    else:
        test.check_result()
    test.check_grad_fn()


input_datas = [
    [torch.tensor(1.5), 0],
    [torch.randn(10), 0],
    [torch.randn(10)[2:8], -1],
    [torch.randn(10, 10), 1],
    [torch.randn(10, 0), 0],
    [torch.randn(10, 10).t(), 1],
    [torch.randn(10, 10, 2, 6, 1, 3), 3],
    [torch.randn(10, 4, 1, 1).to(memory_format=torch.channels_last), 0],
]

dtypes = [
    torch.float16,
    torch.float32,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_median(input_data, dtype):
    input_args = {}
    input_args["input"] = input_data[0].to(dtype)
    dim = input_data[1]
    input_args["dim"] = dim
    test = testing.OpTest(func=torch.median, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
    else:
        test.check_result()
    test.check_grad_fn()


input_datas = [
    [torch.tensor(1.5), 0],
    [torch.randn(10), 0],
    [torch.randn(10)[2:8], -1],
    [torch.randn(10, 10), 1],
    [torch.randn(10, 0), 0],
    [torch.randn(10, 10).t(), 1],
    [torch.randn(10, 10, 2, 6, 1, 3), 3],
    [torch.randn(10, 4, 1, 1).to(memory_format=torch.channels_last), 0],
    [torch.tensor([1, float("nan"), 3, 2]), 0],
    [torch.tensor([[2, 3, 1], [float("nan"), 1, float("nan")]]), 1],
]


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
def test_nanmedian(input_data, dtype):
    input_args = {}
    input_args["input"] = input_data[0].to(dtype)
    dim = input_data[1]
    input_args["dim"] = dim
    test = testing.OpTest(func=torch.nanmedian, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
    else:
        test.check_result()
    test.check_grad_fn()
    test.check_out_ops()


# ----------------------------
# kthvalue.values
# ----------------------------
kthvalue_input_datas = [
    [torch.tensor([1.5]), 0],
    [torch.arange(10, dtype=torch.float32), 0],
    [torch.arange(10, dtype=torch.float32)[2:8], -1],  # slice view
    [torch.arange(10, dtype=torch.float32).view(1, 10).repeat(10, 1), 1],
    # make values strictly increasing along dim=3 (size=6), repeated on other dims
    [
        torch.arange(6, dtype=torch.float32)
        .view(1, 1, 1, 6, 1, 1)
        .expand(10, 10, 2, 6, 1, 3),
        3,
    ],
    [
        torch.arange(10, dtype=torch.float32)
        .view(10, 1, 1, 1)
        .expand(10, 4, 1, 1)
        .to(memory_format=torch.channels_last),
        0,
    ],
]

kthvalue_dtypes = [
    torch.float16,
    torch.float32,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", kthvalue_input_datas)
@pytest.mark.parametrize("dtype", kthvalue_dtypes)
@pytest.mark.parametrize("keepdim", [True, False])
def test_kthvalue_values(input_data, dtype, keepdim):
    x = input_data[0].to(dtype)
    dim = input_data[1]

    # choose a valid k in [1, size_along_dim]
    size = x.size(dim)
    if size == 0:
        pytest.skip("kthvalue requires non-empty dimension")
    k = max(1, size // 2)  # deterministic, always valid

    input_args = {
        "input": x,
        "k": k,
        "dim": dim,
        "keepdim": keepdim,
    }

    test = testing.OpTest(func=torch.kthvalue, input_args=input_args)

    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    else:
        test.check_result()

    # this should cover the out-variant => aten::kthvalue.values
    test.check_out_ops()
    test.check_grad_fn()
