"""
Op Unittest for Attention OP.
"""

# pylint: disable=W0246,E1130,E1102
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

    # pylint: disable=useless-parent-delegation
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def repeak_kv(self, kv, reps):
        batch_size, kv_head_num, seq_len, head_dim = kv.shape
        if reps == 1:
            return kv

        # (bs, kv_head_num, 1, seq_len, head_dim)
        # --> (bs, kv_head_num, reps, seq_len, head_dim)
        # --> (bs, kv_head_num * reps, seq_len, head_dim)
        return (
            kv[:, :, None, :, :]
            .expand(batch_size, kv_head_num, reps, seq_len, head_dim)
            .reshape(batch_size, kv_head_num * reps, seq_len, head_dim)
        )

    def forward(
        self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    ):
        """
        Forward call.
        """
        # query: bs, head_num, seq_len, head_dim
        batch_size, q_head_num, q_seq_len, head_dim = query.shape
        _, kv_head_num, kv_seq_len, _ = key.shape
        assert (
            q_head_num % kv_head_num == 0
        ), "Query's head number must be evenly divided by kv_head_num."

        group_size = q_head_num // kv_head_num
        # For GQA, query must be [bs, kv_head_num * group_size, q_seq_len, head_dim]
        key = self.repeak_kv(key, group_size)
        value = self.repeak_kv(value, group_size)

        half_sqrt = math.sqrt(math.sqrt(head_dim))
        query = query / half_sqrt
        if is_causal:
            kv_seq_len = key.shape[-2]
            # ignore the generated mask for the test case with is_causal=True
            attn_mask = torch.ones(
                (q_seq_len, kv_seq_len), device=query.device, dtype=torch.bool
            ).tril()
        attn_weight = query @ (key.transpose(-2, -1) / half_sqrt)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_mask = torch.zeros_like(
                attn_mask, dtype=query.dtype, device=query.device
            )
            # pylint: disable=invalid-unary-operand-type
            attn_mask = new_mask.masked_fill(~attn_mask, torch.finfo(query.dtype).min)
        if attn_mask is not None and attn_mask.shape == (batch_size, q_seq_len):
            # we should make the mask broadcastable to the atten_probs
            attn_mask = attn_mask.view(batch_size, 1, 1, q_seq_len)
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
            query.dtype
        )

        if dropout_p > 0:
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        # no dropout
        output = attn_weight @ value
        return output


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


def sdp_func(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    batch_size, _, seq_len, _ = query.shape
    if (
        attn_mask is not None
        and attn_mask.shape == (batch_size, seq_len)
        and attn_mask.is_cpu
    ):
        # we should make the mask broadcastable to the atten_probs
        attn_mask = attn_mask.view(batch_size, 1, 1, seq_len)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal
    )


def make_causal_4d_mask_float(
    input_ids_shape, dtype: torch.dtype, device: torch.device = torch.device("cpu")
):
    """
    Make Casual 4D float mask
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


def make_causal_4d_mask_bool(
    input_ids_shape,
    device: torch.device = torch.device("cpu"),
):
    """
    Make Casual 4D bool mask
    """

    bsz, tgt_len = input_ids_shape
    mask = torch.tril(torch.ones((bsz, tgt_len, tgt_len), device=device)).view(
        bsz, 1, tgt_len, tgt_len
    )
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

    kv_num_heads = case[-1]
    num_heads = q_num_heads = case[-2]
    emb_dim = case[-3]
    is_gqa = q_num_heads != kv_num_heads
    if is_gqa:
        assert (
            q_num_heads % kv_num_heads == 0
        ), "Query's head_num should be evenly divided by key/value's head_num."
    assert emb_dim % num_heads == 0  # emb_dim must be evenly divided by num_heads
    head_dim = emb_dim // num_heads
    total_shape = case[0]
    batch_size = total_shape[0]
    seq_len = total_shape[1]
    if not is_gqa:
        total_shape = (batch_size, num_heads, seq_len, 3 * head_dim)
        qkv = torch.randn(total_shape, dtype=dtype)
        # q,k,v has the same shape: [B, num_heads, T, head_dim]
        query, key, value = qkv.chunk(3, -1)  # pylint: disable=invalid-name
    else:
        query = torch.randn([batch_size, q_num_heads, seq_len, head_dim], dtype=dtype)
        key = torch.randn([batch_size, kv_num_heads, seq_len, head_dim], dtype=dtype)
        value = torch.randn([batch_size, kv_num_heads, seq_len, head_dim], dtype=dtype)
    item["query"] = query

    # TODO:(mt-ai):
    # base_test_tool seems having some issues:
    # every input tensor has to be converted to device tensor,
    # which causes totally new tensor created.
    # so we can't check if it is the same tensor of q,k,v in
    # C++ runtime.
    if is_self_attention:
        item["key"] = query
        item["value"] = query
    else:
        item["key"] = key
        item["value"] = value

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
        mask = mask.to(query.dtype)

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
@pytest.mark.parametrize("is_self_attn", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_sdp_backward(case, dtype, func, mask_type, is_self_attn, is_causal):
    """
    Flash SDP test.
    """
    # embedding_dim // q_head_num
    head_dim = case[-3] // case[-2]
    if head_dim not in (64, 128):
        pytest.skip(
            reason="Flash backward doesn't support case with head dim not equal to 64 or 128"
        )
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True):
        input_data = gen_input_data(
            case, mask_type, dtype, is_self_attention=is_self_attn
        )
        input_data["is_causal"] = is_causal
        function(input_data, func, True)
