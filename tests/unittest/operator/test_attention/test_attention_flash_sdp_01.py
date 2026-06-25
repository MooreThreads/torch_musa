"""
Op Unittest for Attention OP.
"""

# pylint: disable=W0246,E1130
import pytest
import torch
import torch.nn.functional as F

from test_attention_base import (
    RawSDP,
    gen_input_data,
    MASK_TYPES,
    sdp_cases,
    sdp_func,
    explicit_scales,
)

from torch_musa import testing
from torch_musa.testing.base_test_tool import DefaultComparator, skip_on_cpu_arch

# FIXME(lgj): dnn current not support headdim=384, dropout not support headdim=[64, 128]
ATTN_HEAD_DIMS = [256, 512]


def function(input_data, func, train=False):
    """
    Test function
    """
    assert isinstance(input_data, dict)
    assert "query" in input_data
    assert "key" in input_data
    assert "value" in input_data

    # FIXME(lms):  mudnn SDP numerical untability, have to set abs_diff=5e-2, rel_diff=1e-3
    # mudnn has: abs_diff=2e-3, rel_diff=2e-3
    comparator = DefaultComparator(abs_diff=5e-2, rel_diff=1e-3)
    refer_func = None
    is_half_or_fp16 = input_data["query"].dtype in {torch.half, torch.bfloat16}
    if is_half_or_fp16:
        refer_func = RawSDP()
    test = testing.OpTest(
        func=func, refer_func=refer_func, input_args=input_data, comparators=comparator
    )
    if train:
        input_data["query"].requires_grad = True
        input_data["key"].requires_grad = True
        input_data["value"].requires_grad = True

    if is_half_or_fp16:
        # CPU doesn't support half.
        test.check_musafp16_vs_musafp16(train=train)
    else:
        # Our reference should use fp32 cpu result
        test.check_result(train=train)


@skip_on_cpu_arch(
    "aarch64", reason="ignore OutOfMemoryError caused by case1 on aarch64"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="SKIP this test if in GPU with arch below 22."
)
@pytest.mark.parametrize("case", sdp_cases(1))
# FIXME:(lms) dtype bfloat16 tensor not supported now
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("func", [sdp_func])
@pytest.mark.parametrize("mask_type", MASK_TYPES)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("explicit_scale", explicit_scales)
def test_flash_sdp(case, dtype, func, mask_type, is_causal, explicit_scale):
    # mask shape flash supports: [B, L], [L,L], [B,1,L,L], [B*Hn, L, L]
    """
    Flash SDP test.
    """
    head_dim = case[-3] // case[-2]
    if testing.get_musa_arch() < 31 and head_dim > 128:
        pytest.skip(
            reason="Flash SDP with head dim > 128 is only supported on arch 31."
        )
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        input_data = gen_input_data(case, mask_type, dtype, is_causal, explicit_scale)
        function(input_data, func)


def _make_qkv(batch, heads, seq_len, head_dim, dtype, device="musa"):
    query = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    key = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    value = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    return query, key, value


def _run_flash(query, key, value, dropout_p, is_causal=False):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="SKIP this test if in GPU with arch below 22."
)
@pytest.mark.parametrize("head_dim", ATTN_HEAD_DIMS)
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_sdp_dropout0_forward_accuracy(head_dim, is_causal):
    """
    Forward with dropout_p=0: flash output must match RawSDP reference.
    """
    if testing.get_musa_arch() < 31:
        pytest.skip(reason="Flash SDP dropout head dims are only supported on arch 31.")
    item = {}
    item["query"] = torch.randn([2, 16, 128, head_dim], dtype=torch.half)
    item["key"] = torch.randn([2, 16, 128, head_dim], dtype=torch.half)
    item["value"] = torch.randn([2, 16, 128, head_dim], dtype=torch.half)
    item["is_causal"] = is_causal
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        function(item, sdp_func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="SKIP this test if in GPU with arch below 22."
)
@pytest.mark.parametrize("head_dim", ATTN_HEAD_DIMS)
@pytest.mark.parametrize("dropout_p", [0.1, 0.3, 0.5])
def test_flash_sdp_dropout_changes_output(head_dim, dropout_p):
    """With dropout_p > 0, output must differ from the p=0 baseline."""
    if testing.get_musa_arch() < 31:
        pytest.skip(reason="Flash SDP dropout head dims are only supported on arch 31.")
    query, key, value = _make_qkv(2, 16, 128, head_dim, torch.half)
    out_p0 = _run_flash(query, key, value, dropout_p=0.0)
    out_dp = _run_flash(query, key, value, dropout_p=dropout_p)

    diff = (out_dp.float() - out_p0.float()).abs().max().item()
    assert diff > 1e-3, f"dropout did not change output (max_diff={diff})"
    assert torch.isfinite(out_dp).all(), "Output contains NaN/Inf"


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="SKIP this test if in GPU with arch below 22."
)
@pytest.mark.parametrize("head_dim", ATTN_HEAD_DIMS)
@pytest.mark.parametrize("dropout_p", [0.1, 0.3])
def test_flash_sdp_dropout_preserves_expectation(head_dim, dropout_p):
    """
    Inverted dropout keeps E[output] close to no-dropout output.
    """
    if testing.get_musa_arch() < 31:
        pytest.skip(reason="Flash SDP dropout head dims are only supported on arch 31.")
    query, key, value = _make_qkv(4, 16, 128, head_dim, torch.half)
    out_p0 = _run_flash(query, key, value, dropout_p=0.0)

    n_trials = 64
    accum = torch.zeros_like(out_p0, dtype=torch.float32)
    for _ in range(n_trials):
        accum += _run_flash(query, key, value, dropout_p=dropout_p).float()
    mean_out = accum / n_trials

    abs_diff = (mean_out - out_p0.float()).abs()
    outlier_ratio = (abs_diff > 0.1).float().mean().item()
    assert outlier_ratio < 1e-3, (
        f"Too many outliers: {outlier_ratio:.4%} exceed atol=0.1, "
        f"max_diff={abs_diff.max().item():.4f}"
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="SKIP this test if in GPU with arch below 22."
)
@pytest.mark.parametrize("head_dim", ATTN_HEAD_DIMS)
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_sdp_dropout_with_causal(head_dim, is_causal):
    """Dropout should work correctly with causal mask."""
    if testing.get_musa_arch() < 31:
        pytest.skip(reason="Flash SDP dropout head dims are only supported on arch 31.")
    query, key, value = _make_qkv(2, 16, 128, head_dim, torch.half)
    out = _run_flash(query, key, value, dropout_p=0.3, is_causal=is_causal)
    assert torch.isfinite(out).all()
    assert out.shape == query.shape
