"""Test fill operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

data_type = testing.get_all_types()
input_data = [
    {"input": torch.rand(5, 3, 2), "value": 10},
    {"input": torch.rand(5, 3, 2), "value": torch.tensor(0.5)},
    {"input": torch.rand(5, 0, 2), "value": 10},
    {"input": torch.rand(5, 0, 2), "value": torch.tensor(127)},
    {"input": torch.rand(0, 0, 0), "value": 10},
    {"input": torch.rand(0, 0, 0), "value": torch.tensor(3)},
    {"input": torch.rand(5, 3, 1, 2, 3), "value": 10},
]

for data in testing.get_raw_data():
    input_data.append({"input": data, "value": 10})


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_fill(input_data, data_type):
    test = testing.OpTest(
        func=torch.fill,
        input_args={
            "input": input_data["input"].to(data_type),
            "value": input_data["value"],
        },
    )
    test.check_result()
    test = testing.InplaceOpChek(
        func_name=torch.fill.__name__ + "_",
        self_tensor=input_data["input"].to(data_type),
        input_args={"value": input_data["value"]},
    )
    test.check_address()
    test.check_res()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("data_type", data_type)
def test_masked_fill(input_data, data_type):
    data = input_data["input"].clone().to(data_type)
    mask = torch.randint(0, 2, data.shape, dtype=torch.bool)
    mask.as_strided_(size=data.shape, stride=data.stride())
    mdata = data.clone().musa()
    mmask = mask.musa()
    value = input_data["value"]
    data.masked_fill_(mask, value)
    mdata.masked_fill_(mmask, value)
    assert torch.allclose(data, mdata.cpu())


def test_masked_fill_large_tensor():
    """Regression test for issue #146: masked fill past 2**31 elements.

    The MUDNN masked-Fill path overflows there; the iterator fallback must
    fill correctly across the boundary. int8 keeps the footprint ~4.4 GiB.
    """
    n = 2**31 + 1024
    try:
        a = torch.zeros(n, dtype=torch.int8, device="musa")
        mask = torch.zeros(n, dtype=torch.bool, device="musa")
        mask[::2] = True
    except Exception as err:  # noqa: BLE001
        if "out of memory" in str(err).lower():
            pytest.skip("insufficient device memory for the >2^31 test")
        raise
    a.masked_fill_(mask, 7)
    # Spot-check the start, the 2^31 boundary, and the tail.
    for lo, hi in ((0, 4), (2**31 - 2, 2**31 + 2), (n - 4, n)):
        got = a[lo:hi].cpu()
        want = torch.tensor([7, 0] * ((hi - lo) // 2), dtype=torch.int8)
        assert torch.equal(got, want)
