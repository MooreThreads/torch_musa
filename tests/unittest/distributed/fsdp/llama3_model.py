"""Llama 3-shaped model used by the FSDP communication tests."""

# pylint: disable=C0103,C0116,E0203
import math

import torch
import torch.nn.functional as F
from torch import nn

# These are the fixed Llama 3-8B architecture parameters.  Only the layer
# count is configurable when constructing the test model.
LLAMA3_DIM = 2048  # 4096 ---> 2048
LLAMA3_HEADS = 32
LLAMA3_KV_HEADS = 8
LLAMA3_VOCAB_SIZE = 128256
LLAMA3_MULTIPLE_OF = 1024
LLAMA3_FFN_MULTIPLIER = 1.3

LLAMA3_TEST_LAYERS = 4
LLAMA3_TEST_SEQ_LEN = 512
LLAMA3_TEST_BATCH_SIZE = 2


def _precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
    )
    positions = torch.arange(end, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    cos, sin = freqs_cis[: x.shape[2]].to(x.device).unbind(-1)
    cos = cos[None, None]
    sin = sin[None, None]
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
    ).flatten(-2)


class _Llama3Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        head_dim = LLAMA3_DIM // LLAMA3_HEADS
        self.n_rep = LLAMA3_HEADS // LLAMA3_KV_HEADS
        self.head_dim = head_dim
        self.wq = nn.Linear(LLAMA3_DIM, LLAMA3_HEADS * head_dim, bias=False)
        self.wk = nn.Linear(LLAMA3_DIM, LLAMA3_KV_HEADS * head_dim, bias=False)
        self.wv = nn.Linear(LLAMA3_DIM, LLAMA3_KV_HEADS * head_dim, bias=False)
        self.wo = nn.Linear(LLAMA3_HEADS * head_dim, LLAMA3_DIM, bias=False)

    def init_weights(self, init_std: float) -> None:
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        query = (
            self.wq(x)
            .view(batch_size, seq_len, LLAMA3_HEADS, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.wk(x)
            .view(batch_size, seq_len, LLAMA3_KV_HEADS, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.wv(x)
            .view(batch_size, seq_len, LLAMA3_KV_HEADS, self.head_dim)
            .transpose(1, 2)
        )
        query = _apply_rotary_emb(query, freqs_cis)
        key = _apply_rotary_emb(key, freqs_cis)
        if self.n_rep != 1:
            key = key.repeat_interleave(self.n_rep, dim=1)
            value = value.repeat_interleave(self.n_rep, dim=1)
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        return self.wo(output.transpose(1, 2).reshape(batch_size, seq_len, -1))


class _Llama3FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_dim = int(2 * (4 * LLAMA3_DIM) / 3)
        hidden_dim = int(LLAMA3_FFN_MULTIPLIER * hidden_dim)
        hidden_dim = LLAMA3_MULTIPLE_OF * math.ceil(hidden_dim / LLAMA3_MULTIPLE_OF)
        self.w1 = nn.Linear(LLAMA3_DIM, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, LLAMA3_DIM, bias=False)
        self.w3 = nn.Linear(LLAMA3_DIM, hidden_dim, bias=False)

    def init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _Llama3Block(nn.Module):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attention_norm = nn.RMSNorm(LLAMA3_DIM, eps=1e-5)
        self.attention = _Llama3Attention()
        self.ffn_norm = nn.RMSNorm(LLAMA3_DIM, eps=1e-5)
        self.feed_forward = _Llama3FeedForward()

    def init_weights(self) -> None:
        init_std = 0.02 / math.sqrt(2 * (self.layer_id + 1))
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attention.init_weights(init_std)
        self.feed_forward.init_weights(init_std)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))


class Llama3Model(nn.Module):
    """A compact Llama 3 model with the production block topology."""

    def __init__(self, num_layers: int, max_seqlen: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_seqlen < 1:
            raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")
        self.max_seqlen = max_seqlen
        self.tok_embeddings = nn.Embedding(LLAMA3_VOCAB_SIZE, LLAMA3_DIM)
        self.layers = nn.ModuleList(
            _Llama3Block(layer_id) for layer_id in range(num_layers)
        )
        self.norm = nn.RMSNorm(LLAMA3_DIM, eps=1e-5)
        self.output = nn.Linear(LLAMA3_DIM, LLAMA3_VOCAB_SIZE, bias=False)
        self.register_buffer(
            "freqs_cis",
            _precompute_freqs_cis(LLAMA3_DIM // LLAMA3_HEADS, max_seqlen),
            persistent=False,
        )

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = _precompute_freqs_cis(
                LLAMA3_DIM // LLAMA3_HEADS, self.max_seqlen
            )
        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers:
            layer.init_weights()
        self.norm.reset_parameters()
        final_out_std = LLAMA3_DIM**-0.5
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-3 * final_out_std,
            b=3 * final_out_std,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(tokens.device)
        for layer in self.layers:
            hidden = layer(hidden, freqs_cis)
        return self.output(self.norm(hidden))


def init_llama3_model(num_layers: int, max_seqlen: int) -> Llama3Model:
    """Create a meta Llama 3-8B-shaped model truncated to ``num_layers``."""
    with torch.device("meta"):
        return Llama3Model(num_layers, max_seqlen)
