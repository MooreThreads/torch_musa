"""Test gather operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

# input and index must have the same number of dimensions
# index.size(d) <= input.size(d)
input_data = [
    {"input": torch.randn(10), "dim": 0, "index": torch.randint(5, (5,))},
    {"input": torch.randn(2, 3), "dim": 1, "index": torch.randint(2, (2, 3))},
    {"input": torch.randn(2, 3, 4), "dim": 2, "index": torch.randint(2, (2, 3, 4))},
    {
        "input": torch.randn(2, 3, 4, 5),
        "dim": 3,
        "index": torch.randint(4, (2, 3, 4, 5)),
    },
    {
        "input": torch.randn(2, 3, 4, 5, 6),
        "dim": 4,
        "index": torch.randint(5, (2, 3, 4, 5, 6)),
    },
    {
        "input": torch.randn(2, 3, 4, 5, 6, 2),
        "dim": 3,
        "index": torch.randint(2, (2, 3, 4, 5, 6, 1)),
    },
    {
        "input": torch.randn(2, 3, 4, 5, 6, 2, 3),
        "dim": 2,
        "index": torch.randint(3, (2, 3, 4, 5, 6, 1, 1)),
    },
    {
        "input": torch.randn(2, 3, 4, 5, 6, 2, 3, 2),
        "dim": 3,
        "index": torch.randint(2, (2, 3, 4, 5, 6, 1, 1, 1)),
    },
]

support_dtypes = [torch.float32, torch.float16]


# TODO(MT-AI): fix error when testing on GPU 1
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_topk_sorted_true(input_data, dtype):
    input_args = {}
    input_args["input"] = input_data["input"].to(dtype)
    input_args["dim"] = input_data["dim"]
    input_args["index"] = input_data["index"]
    test = testing.OpTest(func=torch.gather, input_args=input_args)
    if dtype == torch.float32:
        test.check_result()

    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_take(input_data, dtype):
    input_args = {}
    input_args["input"] = input_data["input"].to(dtype)
    input_args["index"] = input_data["index"]
    test = testing.OpTest(func=torch.take, input_args=input_args)
    if dtype == torch.float32:
        test.check_result()
        test.check_grad_fn()
        test.check_out_ops()
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_put_(dtype):
    input_args = {
        "input": torch.randn(2, 3, 4, 5).to(dtype),
        "index": torch.randint(4, (2, 3, 4, 5)),
        "source": torch.randn(2, 3, 4, 5).to(dtype),
    }

    cpu_input = input_args["input"]
    cpu_index = input_args["index"]
    cpu_source = input_args["source"]

    musa_input = input_args["input"].to("musa")
    musa_index = input_args["index"].to("musa")
    musa_source = input_args["source"].to("musa")

    cpu_input.put_(cpu_index, cpu_source, accumulate=True)
    musa_input.put_(musa_index, musa_source, accumulate=True)

    if dtype == torch.float16:
        abs_diff, rel_diff = 5e-2, 5e-2
    else:
        abs_diff, rel_diff = 1e-5, 1e-6
    cmp = testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff)

    assert cmp(cpu_input, musa_input.cpu())
    assert cpu_input.shape == musa_input.shape
    assert cpu_input.dtype == musa_input.dtype
