"""Test reduce operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import copy
import pytest
import numpy as np
import torch
from torch_musa import testing
from torch_musa.testing import get_musa_arch

input_data = [
    {"input": torch.randn([1, 10]), "dim": 1},
    {"input": torch.randn([1, 10, 5]), "dim": [0, 1]},
    {"input": torch.randn([1, 10, 5, 5]), "dim": 3},
    {
        "input": torch.randn([3, 2, 5, 5]).to(memory_format=torch.channels_last),
        "dim": [1, 3],
    },
    {
        "input": torch.randn([3, 1, 5, 5]).to(memory_format=torch.channels_last),
        "dim": 3,
    },
    {
        "input": torch.randn([3, 4, 1, 1]).to(memory_format=torch.channels_last),
        "dim": [0, -1],
    },
    {"input": torch.randn([1, 10, 5, 5, 10])[..., ::2], "dim": 4},
    {"input": torch.randn([9, 8, 7, 6, 5, 4]), "dim": [1, 3, 4]},
    {"input": torch.randn([9, 8, 7, 6, 5, 4, 16]), "dim": 5},
    {"input": torch.randn([9, 8, 7, 6, 5, 4, 5, 20]), "dim": [0, 2, 4, 6]},
    {"input": torch.randn([9, 8, 7, 6, 5, 4]), "dim": 5},
    {"input": torch.randn([0, 8, 7, 6, 5, 4]), "dim": 5},
    {"input": torch.randn([9, 8, 7, 6, 5, 4, 16]), "dim": 5},
    {"input": torch.randn([9, 8, 7, 6, 5, 4, 5, 20]), "dim": 7},
    {"input": torch.randn([0, 8, 7, 6, 5, 4, 5, 20]), "dim": 7},
]


def function(input_data, dtype, func):
    input_data_cp = copy.deepcopy(input_data)
    if isinstance(input_data_cp["input"], torch.Tensor):
        input_data_cp["input"] = input_data_cp["input"].to(dtype)
    test = testing.OpTest(
        func=func,
        input_args=input_data_cp,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    test.check_result()
    test.check_out_ops()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_amax(input_data, dtype):
    function(input_data, dtype, torch.amax)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_amin(input_data, dtype):
    function(input_data, dtype, torch.amin)


input_data = [
    {"input": torch.randn([1, 10]), "dim": 1},
    {"input": torch.randn([1, 10, 5]), "dim": 2},
    {"input": torch.randn([1, 10, 5, 5]), "dim": 3},
    {
        "input": torch.randn([3, 2, 5, 5]).to(memory_format=torch.channels_last),
        "dim": 1,
    },
    {
        "input": torch.randn([3, 1, 5, 5]).to(memory_format=torch.channels_last),
        "dim": 3,
    },
    {
        "input": torch.randn([3, 4, 1, 1]).to(memory_format=torch.channels_last),
        "dim": 1,
    },
    {"input": torch.randn([1, 10, 5, 5, 10])[..., ::2], "dim": 4},
    {"input": torch.randn([9, 8, 7, 6, 5, 4]), "dim": 5},
    {"input": torch.randn([9, 8, 7, 6, 5, 4, 16]), "dim": 5},
    {"input": torch.randn([9, 8, 7, 6, 5, 4, 5, 20]), "dim": 7},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_mean(input_data, dtype):
    function(input_data, dtype, torch.mean)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sum(input_data, dtype):
    function(input_data, dtype, torch.sum)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config",
    [
        [(15000,), (0)],
        [(182403,), (0)],
        [(242, 342, 52, 2), (3)],
    ],
)
def test_sum_bool(config):
    input_data = {
        "input": torch.randint(low=0, high=2, size=config[0]),
        "dim": config[1],
    }
    function(input_data, torch.bool, torch.sum)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
def test_logsumexp(input_data, dtype):
    function(input_data, dtype, torch.logsumexp)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(get_musa_arch() <= 21, reason="Only support arch greater than 21")
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_logsumexp_low_prec(input_data, dtype):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(
        func=torch.logsumexp,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
        test.check_out_ops(fp16=True)
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16()
        test.check_out_ops(bf16=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_prod(input_data, dtype):
    function(input_data, dtype, torch.prod)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_norm(input_data, dtype):
    function(input_data, dtype, torch.norm)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_norm_fp16(input_data):
    input_data["input"] = input_data["input"].to(torch.float16).to(torch.float32)
    test = testing.OpTest(
        func=torch.norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-2),
    )
    test.check_musafp16_vs_musafp32()
    test.check_out_ops(fp16=True)


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_norm_bf16(input_data):
    input_data["input"] = input_data["input"].to(torch.bfloat16).to(torch.float32)
    test = testing.OpTest(
        func=torch.norm,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=5e-2),
    )
    test.check_musabf16_vs_musafp16()
    test.check_out_ops(bf16=True)


extra_data_for_cumsum = [
    {"input": torch.rand(3, 4) < 0.5, "dim": 1},
    {"input": torch.rand([1, 10, 5]) < 0.5, "dim": 2},
    {"input": torch.rand([1, 10, 5, 5]) < 0.5, "dim": 3},
    {"input": torch.rand([1, 10, 5, 5, 10]) < 0.5, "dim": 4},
    {"input": torch.rand([9, 8, 7, 6, 5, 4]) < 0.5, "dim": 5},
    {"input": torch.rand([9, 8, 7, 6, 5, 4, 16]) < 0.5, "dim": 5},
    {"input": torch.rand([9, 8, 7, 6, 5, 4, 5, 20]) < 0.5, "dim": 7},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data + extra_data_for_cumsum)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cumsum(input_data, dtype):
    function(input_data, dtype, torch.cumsum)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.bool])
def test_any(input_data, dtype):
    function(input_data, dtype, torch.any)


reduce_with_indices_dtype = [
    torch.float32,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.bool,
]
if get_musa_arch() >= 22:
    reduce_with_indices_dtype.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype)
def test_max(input_data, dtype):
    function(input_data, dtype, torch.max)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype)
def test_max_out(input_data, dtype):
    cmp = testing.DefaultComparator()

    def max_fwd(device="cpu"):
        x = input_data["input"].to(device).to(dtype)
        max_values = torch.tensor([], device=device, dtype=dtype)
        indices = torch.tensor([], device=device, dtype=torch.long)
        torch.max(x, input_data["dim"], out=(max_values, indices))

        return max_values, indices

    max_values, indices = max_fwd("cpu")
    max_values_musa, indices_musa = max_fwd("musa")

    cmp(max_values, max_values_musa.cpu())
    cmp(indices, indices_musa.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype)
def test_min(input_data, dtype):
    function(input_data, dtype, torch.min)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.bool])
def test_all(input_data, dtype):
    function(input_data, dtype, torch.all)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype)
def test_argmax(input_data, dtype):
    if dtype == torch.bool:
        pytest.skip(reason="\"argmax_cpu\" not implemented for 'Bool'")
    function(input_data, dtype, torch.argmax)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype)
def test_argmin(input_data, dtype):
    if dtype == torch.bool:
        pytest.skip(reason="\"argmin_cpu\" not implemented for 'Bool'")
    function(input_data, dtype, torch.argmin)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config",
    [
        [
            (0,),
            [0],
        ],
        [
            (450,),
            [0],
        ],
        [
            (2111, 3000),
            [1],
        ],
        [
            (4, 5, 6, 7),
            [0, 2],
        ],
        [
            (2, 3, 4, 5, 6, 7),
            [1, 3, 5],
        ],
        [
            (2, 3, 4, 5, 6, 7, 8, 9),
            [0, 2, 4, 6],
        ],
    ],
)
def test_sum_i32_in_f32_out(config):
    min_val, max_val = -5, 5
    x_np = np.random.uniform(min_val, max_val, size=config[0]).astype("int32")
    x_tensor = torch.from_numpy(x_np)
    test = testing.OpTest(
        func=torch.sum,
        input_args={"input": x_tensor, "dim": config[1], "dtype": torch.float32},
        comparators=testing.DefaultComparator(abs_diff=1e-8),
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config",
    [
        [
            (5,),
            0,
        ],
        [
            (2, 3),
            1,
        ],
        [
            (5, 3, 2),
            1,
        ],
        [
            (2, 2, 5, 2),
            3,
        ],
        [
            (2, 2, 1, 1, 1, 1),
            5,
        ],
    ],
)
@pytest.mark.parametrize("interval", [[-5, 5], [1, 5]])
def test_prod_i32_in_f32_out(config, interval):
    min_val, max_val = interval[0], interval[1]
    x_np = np.random.uniform(min_val, max_val, size=config[0]).astype("int32")
    x_tensor = torch.from_numpy(x_np)
    test = testing.OpTest(
        func=torch.prod,
        input_args={"input": x_tensor, "dim": config[1], "dtype": torch.float32},
        comparators=testing.DefaultComparator(abs_diff=1e-8),
    )
    test.check_result()
    test.check_out_ops()


any_all_integer_input_data = [
    {"input": torch.randint(-1, 1, [1, 10])},
    {"input": torch.randint(-1, 1, [1, 10, 5])},
    {"input": torch.randint(-1, 1, [1, 10, 5, 5])},
    {"input": torch.randint(-1, 1, [1, 10, 5, 5, 10])},
    {"input": torch.randint(-1, 1, [9, 8, 7, 6, 5, 4])},
    {"input": torch.randint(-1, 1, [9, 8, 7, 6, 5, 4, 16])},
    {"input": torch.randint(-1, 1, [9, 8, 7, 6, 5, 4, 5, 20])},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", any_all_integer_input_data)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_any_integer(input_data, dtype):
    input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(
        func=torch.any, input_args=input_data, comparators=testing.BooleanComparator()
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", any_all_integer_input_data)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_all_integer(input_data, dtype):
    input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(
        func=torch.all, input_args=input_data, comparators=testing.BooleanComparator()
    )
    test.check_result()
