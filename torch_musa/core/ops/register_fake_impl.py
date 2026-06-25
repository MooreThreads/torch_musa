"""For operators that lack of FakeTensor implementation,
use torch.library.register_fake to complete registration
"""

from typing import Tuple
import torch


# pylint: disable-all
@torch.library.register_fake("aten::_scaled_dot_product_attention_flash_musa")
def scaled_dot_product_attention_flash_musa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Register FakeTensor impl for aten::_scaled_dot_product_attention_flash_musa"""

    dropout_mask_shape = (0,)

    if query.dim() == 3:
        num_heads, query_seq_len, _ = query.shape
        logsumexp_shape = (num_heads, query_seq_len)
        key_seq_len = key.shape[1]

        if dropout_p > 0.0 and return_debug_mask:
            dropout_mask_shape = (num_heads, query_seq_len, key_seq_len)

    else:
        batch_size, num_heads, query_seq_len, _ = query.shape
        logsumexp_shape = (batch_size, num_heads, query_seq_len)
        key_seq_len = key.shape[2]

        if dropout_p > 0.0 and return_debug_mask:
            dropout_mask_shape = (batch_size, num_heads, query_seq_len, key_seq_len)

    output = query.new_empty(
        (*query.shape[:-3], query.shape[-2], query.shape[-3], value.shape[-1])
    ).transpose(-3, -2)
    logsumexp = query.new_empty(logsumexp_shape, dtype=torch.float32)
    dropout_mask = query.new_empty(dropout_mask_shape, dtype=torch.float32)
    rng_state = torch.empty((2,), dtype=torch.int64, device="cpu")

    return output, logsumexp, dropout_mask, rng_state


@torch.library.register_fake("aten::_scaled_dot_product_attention_flash_musa_backward")
def scaled_dot_product_attention_flash_musa_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    logsumexp: torch.Tensor,
    dropout_mask: torch.Tensor,
    rng_state: torch.Tensor,
    dropout_p: float,
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Register FakeTensor impl for aten::_scaled_dot_product_attention_flash_musa_backward"""

    grad_attn_mask_shape = (0,)
    grad_attn_mask_dtype = torch.float32

    if attn_mask is not None:
        grad_attn_mask_shape = attn_mask.shape
        grad_attn_mask_dtype = attn_mask.dtype

    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    if attn_mask is not None:
        grad_attn_mask = torch.empty_like(attn_mask)
    else:
        grad_attn_mask = query.new_empty(
            grad_attn_mask_shape,
            dtype=grad_attn_mask_dtype,
        )

    return grad_query, grad_key, grad_value, grad_attn_mask
