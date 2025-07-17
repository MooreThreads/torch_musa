"""Test reduce operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import copy
import pytest
import numpy as np
import torch
from torch_musa import testing
from torch_musa.testing import (
    get_musa_arch,
    DefaultComparator,
    AbsDiffComparator,
    BooleanComparator,
)


reduce_dim_configs = [
    # input_shape, dim
    [
        (6,),
        0,
    ],
    [
        (2, 4, 6, 128),
        0,
    ],
    [
        (2, 4, 6, 128),
        -1,
    ],
    [
        (2, 4, 6, 128),
        -1,
    ],
    [
        (2, 4, 6, 128),
        0,
    ],
    [
        (0, 4, 6, 128),
        0,
    ],
    [
        (2, 0, 6, 128),
        0,
    ],
]

reduce_dims_configs = [
    # input_shape, dims
    [
        (2, 4, 6, 128),
        (0,1),
    ],
    [
        (2, 4, 6, 128),
        (-1, 0),
    ],
    [
        (2, 4, 6, 128),
        (-1, 0),
    ],
    [
        (2, 4, 6, 128),
        (0, 2),
    ],
    [
        (0, 4, 6, 128),
        (1, 2),
    ],
    [
        (2, 0, 6, 128),
        (1, 3),
    ],
]

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
    {"input": torch.randn([2, 3, 4]), "dim": None},
]


def function(input_data, input_dtype, func, **kwargs):
    input_data_cp = copy.deepcopy(input_data)
    if isinstance(input_data_cp["input"], torch.Tensor):
        input_data_cp["input"] = input_data_cp["input"].to(input_dtype)
    if kwargs:
        input_data_cp.update(kwargs)
    if input_dtype == torch.float16:
        abs_diff, rel_diff = 1e-3, 1e-3
    elif input_dtype == torch.bfloat16:
        abs_diff, rel_diff = 5e-3, 5e-3
    else:
        abs_diff, rel_diff = 1e-5, 1e-6
    test = testing.OpTest(
        func=func,
        input_args=input_data_cp,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    test.check_result()
    test.check_out_ops()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", testing.get_float_types())
def test_std(input_data, dtype):
    function(input_data, dtype, torch.std)


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


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_aminmax(input_data, dtype):
    if isinstance(input_data["dim"], list):
        return
    function(input_data, dtype, torch.aminmax)


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
    {"input": torch.randn([2, 4, 2, 4, 2, 4]), "dim": 5},
    {"input": torch.randn([2, 4, 2, 4, 2, 4, 16]), "dim": 5},
    {"input": torch.randn([2, 4, 2, 4, 2, 4, 5, 2]), "dim": 7},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_mean(input_data, dtype):
    function(input_data, dtype, torch.mean)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum(input_data, dtype, keepdim):
    function(input_data, dtype, torch.sum, keepdim=keepdim)


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
    function(input_data, torch.bool, torch.sum, dtype=torch.float32)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_logsumexp(input_data, dtype):
    function(input_data, dtype, torch.logsumexp)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(get_musa_arch() <= 21, reason="Only support arch greater than 21")
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_logsumexp_low_prec(input_data, dtype):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    if dtype == torch.float16:
        abs_diff, rel_diff = 5e-3, 1e-3
    else:
        abs_diff, rel_diff = 5e-2, 1e-3
    test = testing.OpTest(
        func=torch.logsumexp,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff, rel_diff),
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


cumulative_float_cases = [
    {
        "input": torch.empty(2, 2, 2).uniform_(),
        "dim": 0,
        "in_dtype": torch.float16,
        "out_dtypes": [torch.float16, torch.float],
    },
    {
        "input": torch.empty(3, 4, 5).uniform_(),
        "dim": 1,
        "in_dtype": torch.bfloat16,
        "out_dtypes": [torch.bfloat16, torch.float],
    },
    {
        "input": torch.empty(2, 2, 2).uniform_(),
        "dim": 2,
        "in_dtype": torch.float,
        "out_dtypes": [torch.float],
    },
]

cumulative_int_cases = [
    {
        "input": torch.randint(-1, 9, (2, 2, 2)),
        "dim": 1,
        "in_dtypes": [torch.int8, torch.short, torch.uint8, torch.int, torch.long],
        "out_dtypes": [torch.int, torch.long],
    },
    {
        "input": torch.rand(2, 2, 2) < 0.9,
        "dim": 1,
        "in_dtypes": [torch.bool],
        "out_dtypes": [torch.int, torch.long],
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("case", cumulative_float_cases)
@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cumulative_float(case, op):
    in_type = case["in_dtype"]
    if in_type == torch.bfloat16 and get_musa_arch() < 22:
        pytest.skip(reason="Not supported")
    out_types = case["out_dtypes"]
    inp, dim = case["input"], case["dim"]
    if in_type != torch.float:
        inp = inp.to(in_type).float()
        comp = DefaultComparator(abs_diff=1e-2, rel_diff=1e-2, equal_nan=True)
    else:
        comp = DefaultComparator(abs_diff=1e-5)
    if in_type != torch.bfloat16:
        cpu_out = op(inp.float(), dim)
    else:
        cpu_out = op(inp.to(in_type), dim).float()

    def do_assert(c_musa, c_cpu, dtype):
        assert c_musa.dtype == dtype
        assert c_musa.shape == c_cpu.shape
        assert comp(c_musa.cpu().float(), c_cpu)

    musa_in = inp.musa().to(in_type)
    musa_out = op(musa_in, dim)
    do_assert(musa_out, cpu_out, in_type)

    for o_t in out_types:
        musa_out = op(musa_in, dim, dtype=o_t)
        do_assert(musa_out, cpu_out, o_t)

        musa_out.zero_()
        op(musa_in, dim, out=musa_out)
        do_assert(musa_out, cpu_out, o_t)

    inplace = getattr(musa_in, op.__name__ + "_")
    inplace(dim)
    do_assert(musa_in, cpu_out, in_type)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("case", cumulative_int_cases)
@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cumulative_int(case, op):
    in_types, out_types = case["in_dtypes"], case["out_dtypes"]
    inp, dim = case["input"], case["dim"]
    comp = AbsDiffComparator(abs_diff=1e-5)

    def do_assert(c_musa, c_cpu, dtype):
        assert c_musa.dtype == dtype
        assert c_musa.shape == c_cpu.shape
        assert comp(c_musa.cpu(), c_cpu)

    for i_t in in_types:
        cpu_in = inp.to(i_t)
        cpu_out = op(cpu_in, dim)  # int64

        musa_in = cpu_in.musa()
        musa_out = op(musa_in, dim)
        do_assert(musa_out, cpu_out, cpu_out.dtype)

        for o_t in out_types:
            musa_out = op(musa_in, dim, dtype=o_t)
            do_assert(musa_out, cpu_out, o_t)

            musa_out.zero_()
            op(musa_in, dim, out=musa_out)
            do_assert(musa_out, cpu_out, o_t)

        if i_t != torch.bool:
            getattr(musa_in, op.__name__ + "_")(dim)
            getattr(cpu_in, op.__name__ + "_")(dim)
            do_assert(musa_in, cpu_in, i_t)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("case", cumulative_float_cases)
@pytest.mark.parametrize("op", [torch.cummin, torch.cummax])
@pytest.mark.skipif(get_musa_arch() < 22, reason="No supported")
def test_cumulative_minmax_float(case, op):
    in_type = case["in_dtype"]
    inp, dim = case["input"], case["dim"]
    if in_type != torch.float:
        comp = DefaultComparator(abs_diff=1e-2, rel_diff=1e-2, equal_nan=True)
    else:
        comp = DefaultComparator(abs_diff=1e-5)
    cpu_out = op(inp, dim)

    def do_assert(c_musa, c_cpu):
        for i in range(2):
            assert c_musa[i].dtype == c_cpu[i].dtype
            assert c_musa[i].shape == c_cpu[i].shape
            assert comp(c_musa[i].cpu(), c_cpu[i])

    musa_in = inp.musa()
    musa_out = op(musa_in, dim)
    do_assert(musa_out, cpu_out)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("case", cumulative_int_cases)
@pytest.mark.parametrize("op", [torch.cummin, torch.cummax])
def test_cumulative_minmax_int(case, op):
    in_types = case["in_dtypes"]
    inp, dim = case["input"], case["dim"]
    comp = AbsDiffComparator(abs_diff=1e-5)
    comp_bool = BooleanComparator()

    def do_assert(c_musa, c_cpu):
        for i in range(2):
            assert c_musa[i].dtype == c_cpu[i].dtype
            assert c_musa[i].shape == c_cpu[i].shape
            if c_cpu[i].dtype == torch.bool:
                assert comp_bool(c_musa[i].cpu(), c_cpu[i])
            else:
                assert comp(c_musa[i].cpu(), c_cpu[i])

    for i_t in in_types:
        cpu_in = inp.to(i_t)
        cpu_out = op(cpu_in, dim)  # int64

        musa_in = cpu_in.musa()
        musa_out = op(musa_in, dim)
        do_assert(musa_out, cpu_out)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.bool])
def test_any(input_data, dtype):
    function(input_data, dtype, torch.any)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dim_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_any_out(config, dtype):
    input_shape, _ = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.any(inp)

    out_musa = torch.empty_like(golden, device="musa")
    result = torch.any(inp.musa(), out=out_musa)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert result.data_ptr() == out_musa.data_ptr()
    assert torch.allclose(golden, result.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dim_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_any_dim(config, dtype):
    input_shape, dim = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.any(inp, dim=dim)
    result = torch.any(inp.musa(), dim=dim)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert torch.allclose(golden, result.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dim_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_any_dim_out(config, dtype):
    input_shape, dim = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.any(inp, dim=dim)

    out_musa = torch.empty_like(golden, device="musa")
    result = torch.any(inp.musa(), dim=dim, out=out_musa)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert result.data_ptr() == out_musa.data_ptr()
    assert torch.allclose(golden, result.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dims_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_any_dims(config, dtype):
    input_shape, dims = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.any(inp, dim=dims)
    result = torch.any(inp.musa(), dim=dims)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert torch.allclose(golden, result.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dims_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_any_dims_out(config, dtype):
    input_shape, dim = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.any(inp, dim=dim)

    out_musa = torch.empty_like(golden, device="musa")
    result = torch.any(inp.musa(), dim=dim, out=out_musa)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert result.data_ptr() == out_musa.data_ptr()
    assert torch.allclose(golden, result.cpu())


reduce_with_indices_dtype = [
    torch.float32,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.int8,
    torch.int16,
]
if get_musa_arch() >= 22:
    reduce_with_indices_dtype.append(torch.bfloat16)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype + [torch.bool])
def test_max(input_data, dtype):
    function(input_data, dtype, torch.max)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype + [torch.bool])
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
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype + [torch.bool])
def test_min(input_data, dtype):
    function(input_data, dtype, torch.min)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", [torch.bool])
def test_all(input_data, dtype):
    function(input_data, dtype, torch.all)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dim_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_all_out(config, dtype):
    input_shape, _ = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.all(inp)

    out_musa = torch.empty_like(golden, device="musa")
    result = torch.all(inp.musa(), out=out_musa)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert result.data_ptr() == out_musa.data_ptr()
    assert torch.allclose(golden, result.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dim_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_all_dim(config, dtype):
    input_shape, dim = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.all(inp, dim=dim)
    result = torch.all(inp.musa(), dim=dim)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert torch.allclose(golden, result.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dims_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_all_dims(config, dtype):
    input_shape, dims = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.all(inp, dim=dims)
    result = torch.all(inp.musa(), dim=dims)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert torch.allclose(golden, result.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dim_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_all_dim_out(config, dtype):
    input_shape, dim = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.all(inp, dim=dim)

    out_musa = torch.empty_like(golden, device="musa")
    result = torch.all(inp.musa(), dim=dim, out=out_musa)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert result.data_ptr() == out_musa.data_ptr()
    assert torch.allclose(golden, result.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", reduce_dims_configs)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.float32])
def test_all_dims_out(config, dtype):
    input_shape, dim = config[0], config[1]
    inp = torch.randint(0, 10, input_shape).to(dtype)
    golden = torch.all(inp, dim=dim)

    out_musa = torch.empty_like(golden, device="musa")
    result = torch.all(inp.musa(), dim=dim, out=out_musa)

    assert golden.dtype == result.dtype
    assert golden.shape == result.shape
    assert result.data_ptr() == out_musa.data_ptr()
    assert torch.allclose(golden, result.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype)
def test_argmax(input_data, dtype):
    function(input_data, dtype, torch.argmax)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.filterwarnings("ignore:An output with one or more elements was resized")
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("resize", [True, False])
@pytest.mark.parametrize("channels_last", [True, False])
def test_argmax_out(keepdim, resize, channels_last):
    input_tensor = torch.randn([1, 10, 5, 5, 10])[..., ::2]  # neither CF nor CL
    dim = 1
    if resize:
        # create new out
        out = torch.randint(0, 10, (1,))
    else:
        out = torch.randint(0, 10, (1, 5, 5, 10))
        if channels_last:
            out = out.to(memory_format=torch.channels_last)
    run_argmax_min_out(torch.argmax, input_tensor, dim, keepdim, out)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", reduce_with_indices_dtype)
def test_argmin(input_data, dtype):
    function(input_data, dtype, torch.argmin)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.filterwarnings("ignore:An output with one or more elements was resized")
@pytest.mark.parametrize("keepdim", [True, False])
def test_argmin_out(keepdim):
    input_tensor = torch.randn([1, 10, 5, 5, 10])[..., ::2]  # neither CF nor CL
    dim = 1
    out = torch.randint(0, 10, (1, 5, 5, 10)).to(memory_format=torch.channels_last)
    run_argmax_min_out(torch.argmin, input_tensor, dim, keepdim, out)


def run_argmax_min_out(op, input_tensor, dim, keepdim, out):
    input_cpu, input_musa = input_tensor.cpu(), input_tensor.musa()
    out_cpu, out_musa = out.cpu(), out.musa()
    result_cpu = op(input_cpu, dim=dim, keepdim=keepdim, out=out_cpu)
    result_musa = op(input_musa, dim=dim, keepdim=keepdim, out=out_musa)

    actual, desired = result_musa.cpu().numpy(), result_cpu.numpy()
    assert (out_cpu.data_ptr() == result_cpu.data_ptr()) == (
        out_musa.data_ptr() == result_musa.data_ptr()
    )
    np.testing.assert_allclose(actual, desired)


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

any_integer_input_data = [
    {"input": torch.randint(-1, 1, [1, 10]), "dim": -1},
    {"input": torch.randint(-1, 1, [1, 10, 5]), "dim": 1},
    {"input": torch.randint(-1, 1, [1, 10, 5, 5]), "dim": 3},
    {"input": torch.randint(-1, 1, [1, 10, 5, 5, 10]), "dim": 4},
    {"input": torch.randint(-1, 1, [9, 8, 7, 6, 5, 4]), "dim": 1},
    {"input": torch.randint(-1, 1, [9, 8, 7, 6, 5, 4, 16]), "dim": 5},
    {"input": torch.randint(-1, 1, [9, 8, 7, 6, 5, 4, 5, 20]), "dim": 7},
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", any_integer_input_data)
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
