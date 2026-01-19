"""Test mm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data_unary = [
    [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
    [torch.randn(8, 1024), torch.randn(64, 128)],
]


@pytest.mark.parametrize("cpu_data", input_data_unary)
@pytest.mark.parametrize(
    "func",
    [
        torch._foreach_abs,
        torch._foreach_acos,
        torch._foreach_asin,
        torch._foreach_atan,
        torch._foreach_ceil,
        torch._foreach_cos,
        torch._foreach_cosh,
        torch._foreach_erf,
        torch._foreach_erfc,
        torch._foreach_exp,
        torch._foreach_expm1,
        torch._foreach_floor,
        torch._foreach_frac,
        torch._foreach_lgamma,
        torch._foreach_log,
        torch._foreach_log10,
        torch._foreach_log1p,
        torch._foreach_log2,
        torch._foreach_max,
        torch._foreach_neg,
        torch._foreach_round,
        torch._foreach_rsqrt,
        torch._foreach_sqrt,
        torch._foreach_sin,
        torch._foreach_sinh,
        torch._foreach_tan,
        torch._foreach_tanh,
        torch._foreach_trunc,
    ],
)
def test_foreach_unary(cpu_data, func):
    musa_data = [t.musa() for t in cpu_data]
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)

    cpu_res = func(cpu_data)
    musa_res = func(musa_data)
    for m_r, c_r in zip(musa_res, cpu_res):
        assert comparator(m_r.cpu(), c_r)

    try:
        inplace_func = getattr(torch, f"{func.__name__}_")
    except AttributeError:
        return
    inplace_func(cpu_data)
    inplace_func(musa_data)
    for m_r, c_r in zip(musa_data, cpu_data):
        assert comparator(m_r.cpu(), c_r)


input_data_binary = [
    {
        "self": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "other": 0.5,
    },
    {
        "self": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "other": [-0.5, 0.0, 0.5],
    },
    {
        "self": [torch.randn(8, 1024), torch.randn(64, 128)],
        "other": torch.tensor(0.5),
    },
    {
        "self": [torch.randn(8, 1024), torch.randn(64, 128)],
        "other": [torch.tensor(-0.5), torch.tensor(0.5)],
    },
]


@pytest.mark.parametrize("input_data", input_data_binary)
@pytest.mark.parametrize(
    "func",
    [
        torch._foreach_add,
        torch._foreach_sub,
        torch._foreach_mul,
        torch._foreach_div,
        torch._foreach_clamp_max,
        torch._foreach_clamp_min,
        # inplace
        torch._foreach_add_,
        torch._foreach_sub_,
        torch._foreach_mul_,
        torch._foreach_div_,
        torch._foreach_clamp_max_,
        torch._foreach_clamp_min_,
    ],
)
def test_foreach_binary(input_data, func):
    cpu_self = input_data["self"]
    cpu_other = input_data["other"]

    musa_self = [t.musa() for t in cpu_self]
    musa_other = None
    if isinstance(cpu_other, list):
        musa_other = []
        for t in cpu_other:
            musa_other.append(t.musa() if isinstance(t, torch.Tensor) else t)
    else:
        musa_other = (
            cpu_other.musa() if isinstance(cpu_other, torch.Tensor) else cpu_other
        )

    cpu_res = func(cpu_self, cpu_other)
    musa_res = func(musa_self, musa_other)

    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
    inplace = func.__name__[-1] == "_"
    if not inplace:
        for m_r, c_r in zip(musa_res, cpu_res):
            assert comparator(m_r.cpu(), c_r)
    else:
        for m_r, c_r in zip(musa_self, cpu_self):
            assert comparator(m_r.cpu(), c_r)


input_data_ternary = [
    {
        "self": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "tensor1": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "tensor2": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "alpha": 0.5,
    },
    {
        "self": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "tensor1": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "tensor2": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "alpha": [-0.5, 0.0, 0.5],
    },
    {
        "self": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "tensor1": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "tensor2": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "alpha": torch.tensor([-0.5, 0.0, 0.5]),
    },
]


@pytest.mark.parametrize("input_data", input_data_ternary)
@pytest.mark.parametrize(
    "func",
    [
        torch._foreach_addcdiv,
        torch._foreach_addcmul,
        # inplace
        torch._foreach_addcdiv_,
        torch._foreach_addcmul_,
    ],
)
def test_foreach_ternary(input_data, func):
    cpu_a = input_data["self"]
    cpu_b = input_data["tensor1"]
    cpu_c = input_data["tensor2"]
    alpha = input_data["alpha"]

    musa_a = [t.musa() for t in cpu_a]
    musa_b = [t.musa() for t in cpu_b]
    musa_c = [t.musa() for t in cpu_c]

    cpu_res = func(cpu_a, cpu_b, cpu_c, alpha)
    musa_res = func(musa_a, musa_b, musa_c, alpha)

    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
    inplace = func.__name__[-1] == "_"
    if not inplace:
        for m_r, c_r in zip(musa_res, cpu_res):
            assert comparator(m_r.cpu(), c_r)
    else:
        for m_r, c_r in zip(musa_a, cpu_a):
            assert comparator(m_r.cpu(), c_r)


# test for foreach_norm
input_data = [
    {"self": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)], "ord": 0},
    {"self": [torch.randn(1024, 2048), torch.randn(64, 4096)], "ord": 2},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_foreach_norm(input_data):
    input_data_musa = {}
    input_data_musa["self"] = []
    for ipt in input_data["self"]:
        input_data_musa["self"].append(ipt.to("musa"))
    input_data_musa["ord"] = input_data["ord"]
    cpu_res = torch._foreach_norm(**input_data)
    musa_res = torch._foreach_norm(**input_data_musa)
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
    print(cpu_res)
    print(musa_res)
    for m_r, c_r in zip(musa_res, cpu_res):
        assert comparator(m_r, c_r)
