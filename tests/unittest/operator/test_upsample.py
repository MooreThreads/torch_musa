"""Test upsampling operators."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
from functools import partial
import pytest
import torch
from torch_musa import testing


all_support_types = [torch.float32]
scale_factor = [2]


def function(input_data, dtype, func):
    if isinstance(input_data["input"], torch.Tensor):
        input_data["input"] = input_data["input"].to(dtype)
    test = testing.OpTest(
        func=func,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6))
    test.check_result()

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data",
[
    {"input": torch.randn([2, 1, 10, 10])},
    {"input": torch.randn([10, 6, 256, 300])},
    {"input": torch.randn([4, 3, 228, 304])},
    {"input": torch.randn([4, 16, 32, 32])}
]
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_bilinear(input_data, dtype, scale_factor, align_corners):
    bilinear = partial(torch.nn.functional.interpolate,
                   mode="bilinear",
                   scale_factor=scale_factor,
                   align_corners=align_corners)
    function(input_data, dtype, bilinear)

    nearest = partial(torch.nn.functional.interpolate,
                   mode="nearest",
                   scale_factor=scale_factor)
    function(input_data, dtype, nearest)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data",
[
    {"input": torch.randn([2, 10, 10])},
    {"input": torch.randn([10, 256, 300])},
    {"input": torch.randn([4, 228, 304])},
    {"input": torch.randn([4, 32, 32])}
]
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_linear(input_data, dtype, scale_factor, align_corners):
    linear = partial(torch.nn.functional.interpolate,
                   mode="linear",
                   scale_factor=scale_factor,
                   align_corners=align_corners)
    function(input_data, dtype, linear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data",
[
    {"input": torch.randn([2, 1, 1, 10, 10])},
    {"input": torch.randn([10, 8, 6, 256, 300])},
    {"input": torch.randn([4, 9, 3, 228, 304])},
    {"input": torch.randn([4, 25, 16, 32, 32])}
]
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_nearest3d(input_data, dtype, scale_factor):
    linear = partial(torch.nn.functional.interpolate,
                   mode="nearest",
                   scale_factor=scale_factor)
    function(input_data, dtype, linear)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data",
[
    {"input": torch.randn([2, 1, 1])},
    {"input": torch.randn([10, 8, 6])},
    {"input": torch.randn([4, 9, 3])},
    {"input": torch.randn([4, 25, 16])}
]
)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("scale_factor", scale_factor)
def test_upsample_nearest1d(input_data, dtype, scale_factor):
    linear = partial(torch.nn.functional.interpolate,
                   mode="nearest",
                   scale_factor=scale_factor)
    function(input_data, dtype, linear)
