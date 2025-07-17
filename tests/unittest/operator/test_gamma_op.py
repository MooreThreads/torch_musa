"""Test lgamma, igamma, digamma, polygamma etc. operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    {
        "input": torch.rand(0),
    },
    {
        "input": torch.rand(5),
    },
    {
        "input": torch.rand(4, 0),
    },
    {
        "input": torch.rand(10, 10),
    },
    {
        "input": torch.rand(2, 256),
    },
    {
        "input": torch.rand(16, 32, 8),
    },
]

float_dtypes = [torch.float32, torch.float16]
# bf16 is not supported on arch older than qy2
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_lgamma(input_data):
    test = testing.OpTest(
        func=torch.lgamma,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_lgamma_(input_data):
    test = testing.InplaceOpChek(
        func_name=torch.lgamma.__name__ + "_",
        self_tensor=input_data["input"],
        comparators=[testing.DefaultComparator(abs_diff=1e-3)],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("float_dtype", float_dtypes)
def test_lgamma_out(input_data, float_dtype):
    input_tensor = input_data["input"].clone().to(float_dtype)
    data = {"input": input_tensor, "out": torch.zeros_like(input_tensor)}
    test = testing.OpTest(
        func=torch.lgamma,
        input_args=data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_digamma(input_data):
    test = testing.OpTest(
        func=torch.digamma,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
def test_digamma_(input_data):
    test = testing.InplaceOpChek(
        func_name=torch.digamma.__name__ + "_",
        self_tensor=input_data["input"],
        comparators=[testing.DefaultComparator(abs_diff=1e-3)],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("float_dtype", float_dtypes)
def test_digamma_out(input_data, float_dtype):
    input_tensor = input_data["input"].clone().to(float_dtype)
    data = {"input": input_tensor, "out": torch.zeros_like(input_tensor)}
    test = testing.OpTest(
        func=torch.digamma,
        input_args=data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()


input_datas_poly = [
    {"input": torch.rand(1), "n": 1},
    {"input": torch.rand(5), "n": 1},
    {"input": torch.rand(4, 1), "n": 2},
    {"input": torch.rand(10, 10), "n": 3},
    {"input": torch.rand(2, 256), "n": 4},
    {"input": torch.rand(16, 32, 8), "n": 5},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas_poly)
def test_polygamma(input_data):
    test = testing.OpTest(
        func=torch.polygamma,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas_poly)
def test_polygamma_out(input_data):
    input_data.update({"out": torch.zeros_like(input_data["input"])})
    test = testing.OpTest(
        func=torch.polygamma,
        input_args=input_data,
        # comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()


input_datas_igamma = [
    {
        "input": torch.rand(1, 5),
        "other": torch.rand(1, 5),
    },
    {
        "other": torch.rand(2, 5),
        "input": torch.rand(2, 5),
    },
    {
        "input": torch.rand(4, 5),
        "other": torch.rand(4, 5),
    },
    {
        "input": torch.rand(10, 10),
        "other": torch.rand(10, 10),
    },
    {
        "input": torch.rand(2, 256),
        "other": torch.rand(2, 256),
    },
    {
        "input": torch.rand(16, 256),
        "other": torch.rand(16, 256),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas_igamma)
def test_igamma(input_data):
    test = testing.OpTest(
        func=torch.igamma,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas_igamma)
def test_igamma_(input_data):
    test = testing.InplaceOpChek(
        func_name=torch.igamma.__name__ + "_",
        self_tensor=input_data["input"],
        input_args={"other": input_data["other"]},
        comparators=[testing.DefaultComparator(abs_diff=1e-3)],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas_igamma)
def test_igamma_out(input_data):
    input_data.update({"out": torch.zeros_like(input_data["input"])})
    test = testing.OpTest(
        func=torch.igammac,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas_igamma)
def test_igammac(input_data):
    test = testing.OpTest(
        func=torch.igammac,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas_igamma)
def test_igammac_(input_data):
    test = testing.InplaceOpChek(
        func_name=torch.igammac.__name__ + "_",
        self_tensor=input_data["input"],
        input_args={"other": input_data["other"]},
        comparators=[testing.DefaultComparator(abs_diff=1e-3)],
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas_igamma)
def test_igammac_out(input_data):
    input_data.update({"out": torch.zeros_like(input_data["input"])})
    test = testing.OpTest(
        func=torch.igammac,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    test.check_result()
