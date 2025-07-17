"""Test functorch vmap features"""

# pylint: disable=unused-variable, missing-function-docstring, unused-argument, unused-import
# pylint: disable=invalid-name, wrong-import-order

import pytest
import torch
import torch_musa

from torch.nn.attention.flex_attention import create_mask
from torch_musa.testing.base_test_tool import BooleanComparator, DefaultComparator

device = "musa"


def test_vmap_in_flex_attention_mask():
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    mask = create_mask(causal_mask, 16, 16, 256, 256, device)

    assert mask.shape == torch.Size([16, 16, 256, 256])
    gold = torch.ones(16, 16, 256, 256, device=device).bool().tril()
    assert BooleanComparator()(mask, gold)


def test_vmap_scalar_in_dim():
    func = torch.dot
    func = torch.vmap(func, in_dims=1)

    a = torch.randn(3, 4, device=device)
    b = torch.randn(3, 4, device=device)
    c = func(a, b)

    assert c.shape == torch.Size([4])
    a = a.transpose(-1, -2).unsqueeze(-2)
    b = b.transpose(-1, -2).unsqueeze(-1)
    gold = (a @ b).squeeze(-1).squeeze(-1)
    assert DefaultComparator()(c, gold)


def test_vmap_scalar_out_dim():
    func = torch.add
    func = torch.vmap(func, out_dims=1)

    a = torch.randn(3, 4, device=device)
    b = torch.randn(3, device=device)
    c = func(a, b)

    assert c.shape == torch.Size([4, 3])
    gold = (a + b.unsqueeze(-1)).transpose(-1, -2)
    assert DefaultComparator()(c, gold)
