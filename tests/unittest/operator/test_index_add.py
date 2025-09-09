"""Test index_add operator."""

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
    [(2, 3, 4, 2, 8193), (3, 2, 4, 2, 8193), 3, 2.0, ((0, 1, 2, 3, 4), (1, 0, 2, 3, 4))],
]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", index_add_coverage_test_config)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_index_add_coverage(config, dtype):
    """logic coverage test index_add, fp32 is enough"""
    add_dim = config[2]

    _input = torch.ones(config[0]).to(dtype)
    _source = torch.randint(0, 4, config[1]).to(dtype)
    if len(config) > 4:
        _input = _input.permute(config[4][0])
        _source = _source.permute(config[4][1])
    input_data = {
        "input": _input,
        "dim": add_dim,
        "index": torch.ones((config[1][add_dim],)).to(torch.int32),
        "source": _source,
        "alpha": config[3],
    }
    # pylint: disable=redefined-builtin
    def func(input, dim, index, source, alpha):
        input.index_add_(dim, index, source, alpha=alpha)
        return input
    test = testing.OpTest(func=func, input_args=input_data)
    test.check_result()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config",
    [
        # self_shape, source_shape, dim, alpha
        [[4, 1024], [6, 1024], 0, 2.0],
        [[16, 128, 2], [16, 128, 2], 1, 1.5],
        [[128, 16, 1024], [128, 8, 1024], 1, 2.5],
    ],
)
@pytest.mark.parametrize("dtype", [
    torch.float32, torch.half, torch.bfloat16, torch.bool,
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
])
def test_index_add(config, dtype):
    """test index_add with supported dtypes"""
    add_dim = config[2]
    input_data = {
        "input": torch.ones(config[0], dtype=dtype),
        "dim": add_dim,
        "index": torch.randint(
            0, config[0][add_dim], (config[1][add_dim],), dtype=torch.int32
        ),
        "source": torch.ones(config[1], dtype=dtype),
        "alpha": config[3],
    }
    if dtype == torch.float16:
        abs_diff, rel_diff = (1e-2, 5e-4)
    elif dtype == torch.bfloat16:
        abs_diff, rel_diff = (5e-2, 5e-3)
    else:
        abs_diff, rel_diff = (1e-6, 1e-6)

    comparator = testing.DefaultComparator(abs_diff, rel_diff, equal_nan=True)
    test = testing.OpTest(func=torch.index_add, input_args=input_data, comparators=comparator)
    test.check_result()


# copy from https://github.com/pytorch/pytorch/blob/v2.0.0/test/test_torch.py#L3966
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("device", ["musa"])
def test_index_add_special(device):
    """test special cases of index_add"""
    shape = (0, 1, 2, 0)
    x = torch.randn(shape, device=device)
    c = x.clone()
    c_clone = c.clone()
    ind_empty = torch.tensor([], dtype=torch.int64, device=device)
    ind_01 = torch.tensor([0, 1], dtype=torch.int64, device=device)

    testing.DefaultComparator()(
        c_clone, c.index_add_(0, ind_empty, torch.empty((0, 1, 2, 0), device=device))
    )
    testing.DefaultComparator()(
        c_clone, c.index_add_(2, ind_empty, torch.empty((0, 1, 0, 0), device=device))
    )
    testing.DefaultComparator()(
        c_clone, c.index_add_(2, ind_01, torch.empty((0, 1, 2, 0), device=device))
    )
