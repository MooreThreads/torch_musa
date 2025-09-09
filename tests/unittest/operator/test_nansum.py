"""Test nansum operator."""

# pylint: disable=missing-function-docstring, unused-import
import random
import pytest
import torch
from torch_musa import testing

torch.set_printoptions(threshold=float("inf"))
# 测试配置：输入形状和求和的维度
configs = [
    # shape, dim
    [(1024,), 0],
    [(4, 256), 1],
    [(4, 256, 2), (0, 2)],  # 多维度求和
    [(4, 256, 2, 2), None],  # 全局求和
    [(4, 1, 20, 20), 2],
    [(0, 3, 4, 5), 2],  # 空张量测试
]


def generate_input(shape, dtype, nan_ratio=0.1):
    """生成包含随机NaN值的输入张量"""
    data = torch.randn(shape, dtype=torch.float32)
    if nan_ratio > 0:
        mask = torch.rand(shape) < nan_ratio
        data[mask] = float("nan")
    return data.to(dtype)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
def test_nansum_fp32(config):
    shape, dim = config
    input_tensor = generate_input(shape, torch.float32, 0.0)
    input_data = {"input": input_tensor, "dim": dim, "keepdim": False}
    test = testing.OpTest(
        func=torch.nansum,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    test.check_result()


@pytest.mark.parametrize("config", configs)
def test_nansum_fp16(config):
    shape, dim = config
    input_data = {
        "input": generate_input(shape, torch.float16, 0.0),
        "dim": dim,
        "keepdim": False,
    }
    test = testing.OpTest(
        func=torch.nansum,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-2),
    )
    test.check_musafp16_vs_musafp32()


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="BF16 requires MUSA arch >= 22"
)
@pytest.mark.parametrize("config", configs)
def test_nansum_bf16(config):
    shape, dim = config
    input_data = {
        "input": generate_input(shape, torch.bfloat16),
        "dim": dim,
        "keepdim": False,
    }
    test = testing.OpTest(
        func=torch.nansum,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=5e-2),
    )
    test.check_result()


def test_nansum_special_cases():
    # 测试全NaN输入
    all_nan = torch.full((4, 4), float("nan"), dtype=torch.float32)
    assert torch.nansum(all_nan) == 0

    # 测试无NaN输入（结果应与普通sum一致）
    no_nan = torch.randn(4, 4)
    assert torch.allclose(torch.nansum(no_nan, dim=1), torch.sum(no_nan, dim=1))

    # 测试keepdim=True
    input_tensor = generate_input((3, 3), torch.float32)
    out = torch.nansum(input_tensor, dim=1, keepdim=True)
    assert out.shape == (3, 1)
