"""Test index_reduce operator."""

import torch
import pytest
from torch_musa import testing

# fmt: off
index_add_coverage_test_config = [
    # self_shape, source_shape, dim, alpha, permute
    # case 0
    [(4, 1024 * 128), (0, 1024 * 128), 0, 2.0],
    [(4, 1024 * 128), (4, 1024 * 128), 0, 2.0],
    [(4, 1024, 127), (4, 1024, 127), 0, 2.0],
    [(1024, 4, 129), (1024, 4, 129), 1, 2.0],
    [(2, 4, 16, 7, 1024), (2, 4, 16, 7, 1024), 3, 2.0, ((1, 0, 2, 3, 4), (1, 0, 2, 3, 4))],
    [(1024, 64, 2, 2), (1024, 64, 2, 3), 3, 2.0, ((1, 0, 2, 3), (1, 0, 2, 3))],
    [(1024, 32, 2, 2, 4), (1024, 32, 2, 2, 4), 4, 2.0, ((3, 2, 1, 0, 4), (3, 2, 1, 0, 4))],
    [(1024, 32, 2, 2, 2), (32, 1024, 2, 2, 2), 4, 2.0, ((0, 1, 2, 3, 4), (1, 0, 2, 3, 4))],
    # case 1
    [(128, 17), (128, 17), 0, 2.0],
    [(128, 2, 22), (128, 4, 22), 1, 2.0],
    [(2, 3, 4, 5), (2, 3, 4, 5), 2, 1.0, ((1, 0, 2, 3), (1, 0, 2, 3))],
    [[1024,], [1024,], 0, 2.0],
    [(3, 8, 1024), (3, 8, 128), 2, 2.0, ((1, 0, 2), (1, 0, 2))],
    [(3, 8, 9, 2), (3, 8, 9, 4), 3, 2.0, ((1, 0, 2, 3), (1, 0, 2, 3))],
    [(4, 4, 9, 2, 14), (4, 4, 9, 2, 15), 4, 2.0, ((0, 1, 2, 3, 4), (1, 0, 2, 3, 4))],
    # case 2
    [(8193, 128), (8193, 128), 0, 1.0],
    [(4, 8193, 9), (4, 8193, 9), 1, 2.0],
    [[8193,], [8193,], 0, 2.0],
    [(2, 2, 2, 16), (2, 2, 2, 8193), 3, 2.0, ((1, 0, 2, 3), (1, 0, 2, 3))],
    [(2, 3, 4, 2, 8193), (3, 2, 4, 2, 8193), 4, 2.0, ((0, 1, 2, 3, 4), (1, 0, 2, 3, 4))],
]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", index_add_coverage_test_config)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("reduce_type", ["prod", "mean", "amin", "amax"])
def test_index_reduce_coverage(config, dtype, reduce_type):
    """logic coverage test index_reduce, fp32 is enough"""
    reduce_dim = config[2]

    _input = torch.ones(config[0]).to(dtype)
    _source = torch.randint(1, 3, config[1]).to(dtype)
    if len(config) > 4:
        _input = _input.permute(config[4][0])
        _source = _source.permute(config[4][1])
    input_data = {
        "input": _input,
        "dim": reduce_dim,
        "index": torch.randint(
            0, config[0][reduce_dim], (config[1][reduce_dim],), dtype=torch.int32
        ),
        "source": _source,
        "reduce": reduce_type,
    }
    # pylint: disable=redefined-builtin
    def func(input, dim, index, source, reduce):
        input.index_reduce_(dim, index, source, reduce)
        return input
    test = testing.OpTest(func=func, input_args=input_data)
    test.check_result()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config",
    [
        # self_shape, source_shape, dim
        [[4, 1024], [6, 1024], 0],
        [[4, 1023], [6, 1023], 0],
        [[16, 128, 2], [16, 128, 2], 1],
        [[128, 16, 1024], [128, 8, 1024], 1],
    ],
)
@pytest.mark.parametrize("dtype", [
    torch.float32, torch.half, torch.bfloat16,
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
])
@pytest.mark.parametrize("reduce_type", ["prod", "mean", "amin", "amax"])
@pytest.mark.parametrize("include_self", [False, True])
def test_index_reduce(config, dtype, reduce_type, include_self):
    """test index_reduce with supported dtypes"""
    reduce_dim = config[2]
    input_data = {
        "input": torch.ones(config[0], dtype=dtype),
        "dim": reduce_dim,
        "index": torch.randint(
            0, config[0][reduce_dim], (config[1][reduce_dim],), dtype=torch.int32
        ),
        "source": torch.randint(1, 3, config[1]).to(dtype),
        "reduce": reduce_type,
        "include_self": include_self
    }
    if dtype == torch.float16:
        abs_diff, rel_diff = (1e-2, 5e-4)
    elif dtype == torch.bfloat16:
        abs_diff, rel_diff = (5e-2, 5e-3)
    else:
        abs_diff, rel_diff = (1e-6, 1e-6)

    comparator = testing.DefaultComparator(abs_diff, rel_diff, equal_nan=True)
    test = testing.OpTest(func=torch.index_reduce, input_args=input_data, comparators=comparator)
    test.check_result()
