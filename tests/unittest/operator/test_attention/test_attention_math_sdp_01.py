"""
Op Unittest for Attention OP.
"""

# pylint: disable=W0246,E1130,E1102
import pytest
import torch

from test_attention_base import RawSDP, gen_input_data, MASK_TYPES, sdp_func

from torch_musa import testing
from torch_musa.testing.base_test_tool import DefaultComparator


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
    return [case for i, case in enumerate(cases) if i % 2 == 1]


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


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="SKIP this test if in GPU with arch below 22."
)
@pytest.mark.parametrize("case", sdp_cases())
# FIXME:(lms) dtype bfloat16 tensor not supported now
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("func", [sdp_func])
@pytest.mark.parametrize("mask_type", MASK_TYPES)
def test_math_sdp(case, dtype, func, mask_type):
    """
    Math SDP test.
    """
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        input_data = gen_input_data(case, mask_type, dtype)
        function(input_data, func)
