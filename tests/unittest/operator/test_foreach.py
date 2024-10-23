"""Test mm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

# test for foreach_mul
input_data = [
    {
        "self": [torch.randn(2, 3), torch.randn(1, 8), torch.randn(10)],
        "other": torch.tensor(1.0),
    },
    {
        "self": [torch.randn(1024, 2048), torch.randn(64, 4096)],
        "other": torch.tensor(1024),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_foreach_mul(input_data):
    input_data_musa = {}
    input_data_musa["self"] = []
    for ipt in input_data["self"]:
        input_data_musa["self"].append(ipt.to("musa"))

    input_data_musa["other"] = input_data["other"].to("musa")
    cpu_res = torch._foreach_mul(**input_data)
    musa_res = torch._foreach_mul(**input_data_musa)
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
    for m_r, c_r in zip(musa_res, cpu_res):
        assert comparator(m_r, c_r)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_foreach_mul_(input_data):
    input_data_musa = {}
    input_data_musa["self"] = []
    for ipt in input_data["self"]:
        input_data_musa["self"].append(ipt.to("musa"))

    input_data_musa["other"] = input_data["other"].to("musa")
    torch._foreach_mul_(**input_data)
    torch._foreach_mul_(**input_data_musa)
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
    for m_r, c_r in zip(input_data_musa["self"], input_data["self"]):
        assert comparator(m_r, c_r)


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
