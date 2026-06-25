"""Test addr operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import copy
import torch
import pytest
import torch_musa

from torch_musa import testing


input_data = [
    {
        "input": torch.randn(4, 4),
        "vec1": torch.randn(4),
        "vec2": torch.randn(4),
        "beta": 1.0,
        "alpha": 1.0,
    },
    {
        "input": torch.randn(4, 4),
        "vec1": torch.randn(4),
        "vec2": torch.randn(4),
        "beta": 0.5,
        "alpha": 2.0,
    },
    {
        # scalar input with outer product
        "input": torch.tensor(0.5),
        "vec1": torch.randn(3),
        "vec2": torch.randn(5),
        "beta": 1.0,
        "alpha": 1.5,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_addr(input_data):
    test = testing.OpTest(
        func=torch.addr,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-3, rel_diff=1e-3),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


inplace_input_data = [
    {
        "input": torch.randn(4, 4),
        "vec1": torch.randn(4),
        "vec2": torch.randn(4),
        "beta": 1.0,
        "alpha": 1.0,
    },
    {
        "input": torch.randn(3, 5),
        "vec1": torch.randn(3),
        "vec2": torch.randn(5),
        "beta": 0.0,
        "alpha": 2.0,
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inplace_input_data)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_addr_inplace(input_data, dtype):
    # use InplaceOpChek to verify addr_ kernel behaviour
    input_data_tmp = copy.deepcopy(input_data)
    self_ = input_data_tmp["input"].to(dtype)
    input_data_tmp["vec1"] = input_data_tmp["vec1"].to(dtype)
    input_data_tmp["vec2"] = input_data_tmp["vec2"].to(dtype)

    input_data_tmp.pop("input")
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
    test = testing.InplaceOpChek(
        func_name=torch.addr.__name__ + "_",
        self_tensor=self_,
        input_args=input_data_tmp,
        comparators=[comparator],
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)


out_input_data = [
    {
        "input": torch.randn(4, 4),
        "vec1": torch.randn(4),
        "vec2": torch.randn(4),
        "beta": 0.3,
        "alpha": 1.7,
    }
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("out_input_data", out_input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_addr_out(out_input_data, dtype):
    cpu_args = copy.deepcopy(out_input_data)
    musa_args = {}
    for key, value in cpu_args.items():
        if isinstance(value, torch.Tensor):
            cpu_args[key] = value.to(dtype)
            musa_args[key] = cpu_args[key].clone().to("musa")
        else:
            musa_args[key] = value

    musa_args["out"] = torch.empty_like(
        torch.addr(
            musa_args["input"], musa_args["vec1"], musa_args["vec2"], beta=0, alpha=1
        )
    )
    cpu_args["out"] = torch.empty_like(
        torch.addr(
            cpu_args["input"], cpu_args["vec1"], cpu_args["vec2"], beta=0, alpha=1
        )
    )

    cpu_res = torch.addr(**cpu_args)
    musa_res = torch.addr(**musa_args)

    comparator = testing.DefaultComparator(abs_diff=1e-3, rel_diff=1e-3)
    assert comparator(cpu_res, musa_res.cpu())
