# pylint: disable=missing-function-docstring,missing-module-docstring
import torch
import pytest
from torch_musa import testing


pool_configs = [
    # input_shape, output_shape, is_channels_last
    # pool 1d cases
    [(1, 64, 8), (5,)],
    [(8, 64, 32), (8,)],
    # pool 2d cases
    [(1, 4, 4), (2, 2), False],
    [(8, 1, 32, 32), (8, 4), False],
    [(8, 1, 32, 32), (8, 4), True],
    [(2, 3, 64, 32), (16, 16), False],
    # pool 3d cases
    [(1, 64, 8, 9, 10), (5, 7, 9)],
    [(1, 64, 8, 9, 10), (5, 7, 9), True],
    [(1, 64, 10, 9, 8), (7, 7, 7)],
]

dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    dtypes.append(torch.bfloat16)

AdaptiveAvgPoolNdMapping = {
    1: torch.nn.AdaptiveAvgPool1d,
    2: torch.nn.AdaptiveAvgPool2d,
    3: torch.nn.AdaptiveAvgPool3d,
}

AdaptiveMaxPoolNdMapping = {
    1: torch.nn.AdaptiveMaxPool1d,
    2: torch.nn.AdaptiveMaxPool2d,
    3: torch.nn.AdaptiveMaxPool3d,
}


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", pool_configs)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("return_indices", [False, True])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("pool_mode", ["MAX", "AVG"])
def test_adaptive_pool(config, dtype, return_indices, train, pool_mode):
    # if pool_mode == "AVG" and testing.get_musa_arch() == 21:
    #     pytest.skip(reason="AdaptiveAvgPool might unstable on S3000")
    if pool_mode == "MAX":
        input_params = {
            "output_size": config[1],
            "return_indices": return_indices,
        }
        func = AdaptiveMaxPoolNdMapping[len(config[1])]
    elif pool_mode == "AVG":
        input_params = {
            "output_size": config[1],
        }
        func = AdaptiveAvgPoolNdMapping[len(config[1])]
    else:
        raise RuntimeError(f"wrong pooling mode: {pool_mode}")
    input_data = torch.randn(config[0]).to(dtype)
    if len(config) > 2 and config[2]:
        memory_format = (
            torch.channels_last_3d if len(config[1]) == 3 else torch.channels_last
        )
        input_data = input_data.to(memory_format=memory_format)
    comparator = testing.DefaultComparator(abs_diff=1e-6, rel_diff=1e-6)
    if dtype == torch.float16:
        comparator = testing.DefaultComparator(abs_diff=1e-3, rel_diff=1e-5)
    if dtype == torch.bfloat16:
        comparator = testing.DefaultComparator(abs_diff=0.016, rel_diff=1e-5)
    test = testing.OpTest(func=func, input_args=input_params, comparators=comparator)
    input_data.requires_grad_(train)
    if dtype == torch.float32:
        test.check_result({"input": input_data}, train=train)
    elif dtype == torch.float16:
        test.check_musafp16_vs_musafp32({"input": input_data}, train=train)
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16({"input": input_data}, train=train)
    test.check_grad_fn()
