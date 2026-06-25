"""Tests for quantized packed matmul ops."""

# pylint: disable=missing-function-docstring, redefined-outer-name
import pytest
import torch

from torch_musa import testing


def _make_valid_int4_qparams(
    k: int, n: int, q_group_size: int, device: str
) -> torch.Tensor:
    groups = k // q_group_size
    return torch.randn((groups, n, 2), dtype=torch.bfloat16, device=device)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inner_k_tiles", [2, 4, 8])
def test_convert_weight_to_int4pack_success(inner_k_tiles):
    n = 16
    packed_k = inner_k_tiles * 8 * 4
    weight = torch.randint(0, 256, (n, packed_k), dtype=torch.uint8, device="musa")

    out = torch.ops.aten._convert_weight_to_int4pack(weight, inner_k_tiles)

    assert out.dtype == torch.int32
    assert out.device.type == "musa"
    assert out.dim() == 4
    assert out.size(0) == (n + 7) // 8
    assert out.size(1) == packed_k // (inner_k_tiles * 8)
    assert out.size(2) == 32
    assert out.size(3) == inner_k_tiles // 2


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_convert_weight_to_int4pack_invalid_args():
    n = 8
    packed_k = 32
    good = torch.randint(0, 256, (n, packed_k), dtype=torch.uint8, device="musa")

    with pytest.raises(RuntimeError):
        torch.ops.aten._convert_weight_to_int4pack(good.to(torch.int32), 2)

    with pytest.raises(RuntimeError):
        torch.ops.aten._convert_weight_to_int4pack(good.unsqueeze(0), 2)

    with pytest.raises(RuntimeError):
        torch.ops.aten._convert_weight_to_int4pack(good.t(), 2)

    with pytest.raises(RuntimeError):
        torch.ops.aten._convert_weight_to_int4pack(good, 3)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() <= 22,
    reason="bfloat16/int4 packed mm requires PH1 or later",
)
@pytest.mark.parametrize("inner_k_tiles", [2, 4, 8])
@pytest.mark.parametrize("q_group_size", [32, 64])
def test_weight_int4pack_mm_success(inner_k_tiles, q_group_size):
    if inner_k_tiles == 8 and q_group_size == 32:
        pytest.skip("inner_k_tiles=8 is not compatible with q_group_size=32")
    m = 8
    n = 16
    k = q_group_size * 2
    packed_k = k // 2

    act = torch.randn((m, k), dtype=torch.bfloat16, device="musa")
    raw_weight = torch.randint(0, 256, (n, packed_k), dtype=torch.uint8, device="musa")
    packed_weight = torch.ops.aten._convert_weight_to_int4pack(
        raw_weight, inner_k_tiles
    )
    q_scale_and_zeros = _make_valid_int4_qparams(k, n, q_group_size, device="musa")

    out = torch.ops.aten._weight_int4pack_mm(
        act, packed_weight, q_group_size, q_scale_and_zeros
    )

    assert out.dtype == torch.bfloat16
    assert out.device.type == "musa"
    assert tuple(out.shape) == (m, n)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="bfloat16/int4 packed mm requires QY2 or later",
)
def test_weight_int4pack_mm_invalid_args():
    m = 4
    n = 8
    q_group_size = 32
    k = 64
    packed_k = k // 2

    act = torch.randn((m, k), dtype=torch.bfloat16, device="musa")
    raw_weight = torch.randint(0, 256, (n, packed_k), dtype=torch.uint8, device="musa")
    packed_weight = torch.ops.aten._convert_weight_to_int4pack(raw_weight, 2)
    q_scale_and_zeros = _make_valid_int4_qparams(k, n, q_group_size, device="musa")

    with pytest.raises(RuntimeError):
        torch.ops.aten._weight_int4pack_mm(
            act.float(), packed_weight, q_group_size, q_scale_and_zeros
        )

    with pytest.raises(RuntimeError):
        torch.ops.aten._weight_int4pack_mm(act, packed_weight, 16, q_scale_and_zeros)

    with pytest.raises(RuntimeError):
        torch.ops.aten._weight_int4pack_mm(
            act, packed_weight, q_group_size, q_scale_and_zeros[..., :1]
        )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_weight_int8pack_mm_success(dtype):
    m = 16
    n = 12
    k = 32
    x = torch.randn((m, k), dtype=dtype, device="musa")
    w_int8 = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="musa")
    scales = torch.rand((n,), dtype=torch.float32, device="musa") + 1e-2

    out = torch.ops.aten._weight_int8pack_mm(x, w_int8, scales)
    assert out.device.type == "musa"
    assert out.dtype == torch.float32
    assert tuple(out.shape) == (m, n)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_weight_int8pack_mm_zero_input_numeric():
    m = 8
    n = 10
    k = 16
    x = torch.zeros((m, k), dtype=torch.float32, device="musa")
    w_int8 = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="musa")
    scales = torch.ones((n,), dtype=torch.float32, device="musa")

    out = torch.ops.aten._weight_int8pack_mm(x, w_int8, scales)
    comparator = testing.DefaultComparator(abs_diff=1e-6, rel_diff=1e-6)
    assert comparator(out.cpu(), torch.zeros((m, n), dtype=torch.float32))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_weight_int8pack_mm_invalid_args():
    m = 8
    n = 10
    k = 16
    x = torch.randn((m, k), dtype=torch.float32, device="musa")
    w_int8 = torch.randint(-128, 127, (n, k), dtype=torch.int8, device="musa")
    scales = torch.rand((n,), dtype=torch.float32, device="musa") + 1e-2

    with pytest.raises(RuntimeError):
        torch.ops.aten._weight_int8pack_mm(x.unsqueeze(0), w_int8, scales)

    with pytest.raises(RuntimeError):
        torch.ops.aten._weight_int8pack_mm(x, w_int8.t(), scales)

    with pytest.raises(RuntimeError):
        torch.ops.aten._weight_int8pack_mm(x, w_int8, scales[:-1])
