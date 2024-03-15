# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import, not-callable
import torch
import pytest
from torch_musa import testing

import torch_musa


input_dtype = [torch.float32]

parameter = [
    {"data": torch.randn(6, 6), "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(6, 6, 6), "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(20, 6, 10, 10), "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(20, 6, 10, 1, 10),
     "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(20, 6, 10, 2, 1, 10),
     "num_groups": 3, "num_channels": 6},
    {"data": torch.randn(20, 6, 10, 10, 1, 2, 3),
     "num_groups": 3, "num_channels": 6},
    {
        "data": torch.randn(20, 6, 10, 10, 1, 2, 3, 1),
        "num_groups": 3,
        "num_channels": 6,
    },
]

affine = [True]

eps = [1e-5, 0, 0.5]

train = [True, False]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_dtype", input_dtype)
@pytest.mark.parametrize("parameter", parameter)
@pytest.mark.parametrize("affine", affine)
@pytest.mark.parametrize("eps", eps)
@pytest.mark.parametrize("train", train)
def test_native_group_norm(input_dtype, parameter, affine, eps, train):
    test = testing.OpTest(
        func=torch.nn.GroupNorm,
        input_args={
            "num_groups": parameter["num_groups"],
            "num_channels": parameter["num_channels"],
            "eps": eps,
            "affine": affine,
        },
        # NB: this test may fail with abs-diff=1e-6
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result(
        inputs={
            "input": torch.tensor(
                parameter["data"].detach().numpy(),
                dtype=input_dtype,
                requires_grad=True,
            )
        },
        train=train,
    )


@testing.skip_if_not_multiple_musa_device
def test_native_group_norm_device():
    data = torch.randn(6, 6, 6, requires_grad=True)
    num_groups = 3
    num_channels = 6
    affine = True
    eps = 0.5
    group_norm = torch.nn.GroupNorm(num_groups, num_channels, eps, affine)
    group_norm(data).sum().backward()

    musa_data = torch.tensor(data.detach().numpy(),
                             requires_grad=True, device="musa:1")
    musa_group_norm = group_norm.to("musa:1")
    musa_group_norm(musa_data).sum().backward()

    assert testing.DefaultComparator(1e-5)(musa_data.grad.cpu(), data.grad)
    assert musa_data.grad.shape == data.grad.shape
    assert musa_data.grad.dtype == data.grad.dtype


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="fp16 group_norm supported in S4000 or later"
)
@pytest.mark.parametrize("input_dtype", input_dtype)
@pytest.mark.parametrize("parameter", parameter)
@pytest.mark.parametrize("affine", affine)
@pytest.mark.parametrize("eps", eps)
@pytest.mark.parametrize("train", train)
def test_native_group_norm_fp16(input_dtype, parameter, affine, eps, train):
    m = torch.nn.GroupNorm(
        num_groups=parameter["num_groups"],
        num_channels=parameter["num_channels"],
        eps=eps,
        affine=affine,
    )
    m.train(train)
    input_data = torch.tensor(
        parameter["data"].detach().numpy(), dtype=input_dtype, requires_grad=True
    )
    output = m(input_data)
    m.half()
    input_data = input_data.half()
    musa_output = m.to("musa")(input_data.to("musa"))
    assert testing.DefaultComparator(
        abs_diff=1e-2)(output, musa_output.cpu().float())
