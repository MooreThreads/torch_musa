"""
Op Unittest for Attention OP.
"""
import math

import pytest
import torch
from torch.nn import functional as F
from torch_musa import testing
from torch_musa.testing.base_test_tool import DefaultComparator


# NOTE(yutian): 4d mask, 2: 4d float, 3: 4d bool
# FIXME(yutian) Check the mask type equals to 4
MASK_TYPES = [-1, 0, 1, 2]
# ============ Below are only for ScaledDotProductAttention Tests. ===========


class RawSDP(torch.nn.Module):
    """
        This is a demo hand-maded SDP operation like
        `torch.nn.functional.scaled_dot_product_attention` did.
    """

    def __init__(self, *args, **kwargs) -> None:  # pylint: disable=useless-parent-delegation
        super().__init__(*args, **kwargs)

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_casual=False):
        """
            Forward call.
        """
        # query: bs, head_num, seq_len, head_dim
        batch_size, _, q_seq_len, head_dim = query.shape
        half_sqrt = math.sqrt(math.sqrt(head_dim))
        query = query / half_sqrt
        if is_casual:
            assert attn_mask is not None
            kv_seq_len = key.shape[-2]
            attn_mask = torch.ones(
                (q_seq_len, kv_seq_len), device=query.device, dtype=torch.bool).tril()
        attn_weight = query @ (key.transpose(-2, -1) / half_sqrt)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_mask = torch.zeros_like(
                attn_mask, dtype=query.dtype, device=query.device)
            attn_mask = new_mask.masked_fill(
                ~attn_mask, torch.finfo(query.dtype).min)  # pylint: disable=invalid-unary-operand-type

        if attn_mask is not None and attn_mask.shape == (batch_size, q_seq_len):
            # we should make the mask broadcastable to the atten_probs
            attn_mask = attn_mask.view(batch_size, 1, 1, q_seq_len)
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        attn_weight = torch.softmax(
            attn_weight, dim=-1, dtype=torch.float32).to(query.dtype)

        if dropout_p > 0:
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        # no dropout
        output = attn_weight @ value
        return output


def sdp_cases():
    """
    Return atten cases
    """
    cases = [
        [(2, 4, 6), 6, 2],
        [(1, 32, 128), 128, 4],
        [(4, 32, 128), 128, 4],
        [(16, 32, 128), 128, 4],
        [(128, 32, 1024), 1024, 16]
    ]
#    device_arch_name = torch.musa.get_device_properties(
#        torch.musa.current_device()).name
#    if device_arch_name != "MTT S80":
#        cases.extend([
#            [(2, 512, 2048), 2048, 32],
#            [(2, 1024, 4096), 4096, 32],
#            [(4, 2048, 2048), 2048, 16],
#            [(1, 4096, 8192), 8192, 64],
#            [(1, 4096, 4096), 4096, 32],
#            [(1, 4096, 2048), 2048, 16],
#            [(2, 4096, 1024), 1024, 8],
#            [(2, 4096, 8192), 8192, 64],
#            [(2, 4096, 4096), 4096, 32],
#            [(2, 4096, 2048), 2048, 16],
#            [(2, 4096, 1024), 1024, 8],
#            [(1, 2048, 1024), 1024, 8],
            # [(30, 2048, 4096), 4096, 32]
            # [(1, 4096, 6656), 6656, 52]
#        ])
    return cases


def sdp_func(query, key, value, attn_mask=None, dropout_p=0.0, is_casual=False):
    batch_size, _, seq_len, _ = query.shape
    if attn_mask is not None and attn_mask.shape == (batch_size, seq_len) and attn_mask.is_cpu:
        # we should make the mask broadcastable to the atten_probs
        attn_mask = attn_mask.view(batch_size, 1, 1, seq_len)
    return F.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_casual)


def make_causal_4d_mask_float(
    input_ids_shape,
    dtype: torch.dtype,
    device: torch.device = torch.device("cpu")
):
    """
    Make Casual 4D float mask
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len),
                      torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len
    )


def make_causal_4d_mask_bool(
    input_ids_shape,
    device: torch.device = torch.device("cpu"),
):
    """
    Make Casual 4D bool mask
    """

    bsz, tgt_len = input_ids_shape
    mask = torch.tril(torch.ones(
        (bsz, tgt_len, tgt_len), device=device)).view(
            bsz, 1, tgt_len, tgt_len)
    mask = mask > 0.5

    return mask


def generate_square_subsequent_mask(seq_len: int):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def generate_pad_mask(batch_size: int, seq_len: int):
    mask = torch.zeros([batch_size, seq_len], dtype=torch.float)
    for b in range(batch_size):
        r = torch.randint(1, seq_len - 1, (1,))
        mask[b][-r:] = -torch.inf
    return mask


def gen_input_data(case, mask_type, dtype=torch.float32, is_self_attention=False):
    """
        Generating the mocked input data of SDP.
    """
    item = {}

    num_heads = case[-1]
    emb_dim = case[-2]
    assert emb_dim % num_heads == 0  # emb_dim must be evenly divided by num_heads
    head_dim = emb_dim // num_heads
    total_shape = case[0]
    batch_size = total_shape[0]
    seq_len = total_shape[1]
    total_shape = (batch_size, num_heads, seq_len, 3 * head_dim)
    qkv = torch.randn(total_shape, dtype=dtype)
    # q,k,v has the same shape: [B, num_heads, T, head_dim]
    q, k, v = qkv.chunk(3, -1)  # pylint: disable=invalid-name
    item["query"] = q

    # TODO:(mt-ai):
    # base_test_tool seems having some issues:
    # every input tensor has to be converted to device tensor,
    # which causes totally new tensor created.
    # so we can't check if it is the same tensor of q,k,v in
    # C++ runtime.
    if is_self_attention:
        item["key"] = q
        item["value"] = q
    else:
        item["key"] = k
        item["value"] = v

    # generating bool mask.
    if mask_type == 1:
        # padding mask
        mask = generate_pad_mask(batch_size, seq_len)
    elif mask_type == 0:
        # key padding
        mask = generate_square_subsequent_mask(seq_len)
    elif mask_type == 2:
        mask = make_causal_4d_mask_float((batch_size, seq_len), dtype=dtype)
    elif mask_type == 3:
        mask = make_causal_4d_mask_bool((batch_size, seq_len))
    elif mask_type == 4:
        mask = ~make_causal_4d_mask_bool((batch_size, seq_len))
    else:
        mask = None

    if mask is not None and mask.dtype != torch.bool:
        mask = mask.to(q.dtype)

    item["attn_mask"] = mask

    return item


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
    test = testing.OpTest(func=func, refer_func=refer_func,
                          input_args=input_data, comparators=comparator)
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
@pytest.mark.parametrize("case", sdp_cases())
# FIXME:(lms) dtype bfloat16 tensor not supported now
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("func", [sdp_func])
@pytest.mark.parametrize("mask_type", MASK_TYPES)
@pytest.mark.parametrize("is_self_attn", [True, False])
def test_math_sdp(case, dtype, func, mask_type, is_self_attn):
    """
    Math SDP test.
    """
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False):
        input_data = gen_input_data(
            case, mask_type, dtype, is_self_attention=is_self_attn)
        function(input_data, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("case", sdp_cases())
@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
@pytest.mark.parametrize("func", [sdp_func])
@pytest.mark.parametrize("mask_type", MASK_TYPES)
@pytest.mark.parametrize("is_self_attn", [False])
def test_math_sdp_backward(case, dtype, func, mask_type, is_self_attn):
    """
    Math SDP backward test.
    """
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False):
        input_data = gen_input_data(
            case, mask_type, dtype, is_self_attention=is_self_attn)
        function(input_data, func, True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(f"{torch.musa.get_device_properties(torch.musa.current_device()).major}"
                    f".{torch.musa.get_device_properties(torch.musa.current_device()).minor}"
                    < "2.2", reason="SKIP this test if in GPU with arch below 2.2(S4000).")
@pytest.mark.parametrize("case", sdp_cases())
# FIXME:(lms) dtype bfloat16 tensor not supported now
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("func", [sdp_func])
@pytest.mark.parametrize("mask_type", MASK_TYPES)
@pytest.mark.parametrize("is_self_attn", [True, False])
def test_flash_sdp(case, dtype, func, mask_type, is_self_attn):
    """
    Flash SDP test.
    """
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True):
        input_data = gen_input_data(
            case, mask_type, dtype, is_self_attention=is_self_attn)
        function(input_data, func)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(f"{torch.musa.get_device_properties(torch.musa.current_device()).major}."
                    f"{torch.musa.get_device_properties(torch.musa.current_device()).minor}"
                    < "2.2", reason="SKIP this test if in GPU with arch below 2.2(S4000).")
@pytest.mark.parametrize("case", sdp_cases())
# FIXME:(lms) dtype bfloat16 tensor not supported now
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("func", [sdp_func])
@pytest.mark.parametrize("mask_type", MASK_TYPES)
@pytest.mark.parametrize("is_self_attn", [True, False])
def test_flash_sdp_backward(case, dtype, func, mask_type, is_self_attn):
    """
    Flash SDP test.
    """
    head_dim = case[-2]/case[-1]
    if head_dim not in (64, 128):
        pytest.skip(
            reason="Flash backward doesn't support case with head dim not equal to 64 or 128")
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True):
        input_data = gen_input_data(
            case, mask_type, dtype, is_self_attention=is_self_attn)
        function(input_data, func, True)
