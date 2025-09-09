"""
Op Unittest for Attention OP.
"""

# pylint: disable=W0246,E1130
import pytest
import torch

from test_attention_base import (
    RawSDP,
    gen_input_data,
    MASK_TYPES,
    sdp_func,
    explicit_scales,
    make_causal_4d_mask_float,
)

from torch_musa import testing
from torch_musa.testing.base_test_tool import DefaultComparator, skip_on_cpu_arch


def sdp_cases():
    """
    Return atten cases
    """
    # [(bs, seq_len, embedding_dim), embedding_dim, q_head_num, kv_head_num]
    cases = [
        [(128, 32, 1024), 1024, 16, 16],
        # gqa
        [(128, 32, 1024), 1024, 16, 4],
    ]

    device_arch_name = torch.musa.get_device_properties(
        torch.musa.current_device()
    ).name
    if device_arch_name != "MTT S80":
        cases.extend(
            [
                [(2, 512, 2048), 2048, 32, 32],
                [(2, 4096, 4096), 4096, 32, 32],
                [(1, 2048, 1024), 1024, 8, 8],
            ]
        )
    return [case for i, case in enumerate(cases) if i % 2 == 0]


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
    comparator = DefaultComparator(abs_diff=1.3e-1, rel_diff=1e-3)
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


@skip_on_cpu_arch("aarch64", reason="RuntimeError remains to be tackled on aarch64")
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="SKIP this test if in GPU with arch below 22."
)
@pytest.mark.parametrize("case", sdp_cases())
# FIXME:(lms) dtype bfloat16 tensor not supported now
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("func", [sdp_func])
@pytest.mark.parametrize("mask_type", MASK_TYPES)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("explicit_scale", explicit_scales)
def test_flash_sdp_backward(case, dtype, func, mask_type, is_causal, explicit_scale):
    """
    Flash SDP test.
    """
    # embedding_dim // q_head_num
    head_dim = case[-3] // case[-2]
    if head_dim not in (64, 128):
        pytest.skip(
            reason="Flash backward doesn't support case with head dim not equal to 64 or 128"
        )
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        input_data = gen_input_data(case, mask_type, dtype, is_causal, explicit_scale)
        function(input_data, func, True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="SKIP this test if in GPU with arch below 22."
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("func", [sdp_func])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_sdp_dim3_backward(dtype, func, is_causal):
    """
    Flash SDP with dim3 q/k/v.
    """
    item = {}
    item["query"] = torch.randn([4, 512, 64], dtype=dtype)
    item["key"] = torch.randn([4, 512, 64], dtype=dtype)
    item["value"] = torch.randn([4, 512, 64], dtype=dtype)
    if is_causal:
        item["is_causal"] = True
    else:
        dim3_mask = make_causal_4d_mask_float((1, 512), dtype)
        item["attn_mask"] = torch.squeeze(dim3_mask, 0)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        function(item, func, True)
