"""
Op Unittest for Attention OP.
"""

# pylint: disable=C0116, C0103, C0115, R1704
import copy

import torch
import pytest

parametrize = pytest.mark.parametrize


class TestMHA:
    def _test_multihead_attention_impl(
        self,
        dtype,
        mode,
        use_nt,
        need_weights,
        average_attn_weights,
        use_padding=False,
        pad_all=False,
    ):
        device = "musa"
        embed_dim = 512
        num_heads = 4
        bs = 1
        sl = 512

        q = 6 * torch.rand(bs, sl, embed_dim, device=device, dtype=torch.float32) - 3
        if use_padding:
            if pad_all:
                for q_i in q:
                    q_i[-1] = torch.zeros_like(
                        q[0][-1], device=device, dtype=torch.float32
                    )
                mask = torch.zeros(q.shape[:-1], device=device, dtype=torch.bool)
                for mask_i in mask:
                    mask_i[-1] = True
            else:
                q[0][-1] = torch.zeros_like(
                    q[0][-1], device=device, dtype=torch.float32
                )
                mask = torch.zeros(q.shape[:-1], device=device, dtype=torch.bool)
                mask[0][-1] = True
        if mode == "self":
            k = q
            v = q
        elif mode == "encdec":
            k = (
                6 * torch.rand(bs, sl, embed_dim, device=device, dtype=torch.float32)
                - 3
            )
            v = k
        elif mode == "generic":
            k = (
                6 * torch.rand(bs, sl, embed_dim, device=device, dtype=torch.float32)
                - 3
            )
            v = (
                6 * torch.rand(bs, sl, embed_dim, device=device, dtype=torch.float32)
                - 3
            )
        else:
            self.fail(f"invalid mode `{mode}`!")

        qkv = torch.nn.Linear(
            embed_dim, 3 * embed_dim, device=device, dtype=torch.float32
        )
        native_qkv = copy.deepcopy(qkv).to(dtype=dtype)

        proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=torch.float32)
        native_proj = copy.deepcopy(proj).to(dtype=dtype)

        pt = torch.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, device=device, dtype=torch.float32
        )

        pt.in_proj_weight = qkv.weight
        pt.in_proj_bias = qkv.bias
        pt.out_proj.weight = proj.weight
        pt.out_proj.bias = proj.bias
        pt = pt.to(dtype)
        pt = pt.to(torch.float32)

        class NativeMHA(torch.nn.Module):
            def __init__(self, embed_dim, num_heads, qkv, proj):
                super().__init__()
                self.qkv = qkv
                self.proj = proj
                self.embed_dim = embed_dim
                self.num_heads = num_heads

            def forward(self, q, k, v, key_padding_mask):
                return torch._native_multi_head_attention(
                    q,
                    k,
                    v,
                    self.embed_dim,
                    self.num_heads,
                    self.qkv.weight,
                    self.qkv.bias,
                    self.proj.weight,
                    self.proj.bias,
                    key_padding_mask,
                    need_weights=need_weights,
                    average_attn_weights=average_attn_weights,
                    mask_type=1,  # mask_type = 1 => src_key_padding_mask, mask_type = 0 => src_mask
                )

        npt = NativeMHA(
            embed_dim=embed_dim, num_heads=num_heads, qkv=native_qkv, proj=native_proj
        ).to(dtype)

        pt = pt.to(device)
        # pt.eval()
        npt = npt.to(device)
        # npt.eval()
        ypt, weight_pt = pt(
            q,
            k,
            v,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            key_padding_mask=mask if use_padding else None,
        )
        if use_nt:
            qs = list(torch.unbind(q))
            if use_padding:
                if pad_all:
                    qs = [x[:-1] for x in qs]
                else:
                    qs[0] = qs[0][:-1]
            q = torch.nested.nested_tensor(qs, device=device, dtype=dtype)
            if mode == "self":
                k = v = q
            elif mode == "encdec":
                k = torch.nested.nested_tensor(
                    torch.unbind(k), device=device, dtype=dtype
                )
                v = k
            else:
                k = torch.nested.nested_tensor(
                    torch.unbind(k), device=device, dtype=dtype
                )
                v = torch.nested.nested_tensor(
                    torch.unbind(v), device=device, dtype=dtype
                )

        native_q = q.to(dtype=dtype)
        native_k = k.to(dtype=dtype)
        native_v = v.to(dtype=dtype)

        ynpt, weight_npt = npt(
            native_q,
            native_k,
            native_v,
            key_padding_mask=mask if use_padding and not use_nt else None,
        )
        if use_nt:
            ynpt = ynpt.to_padded_tensor(0)
            if pad_all:
                ynpt_final = torch.zeros_like(ypt)
                ynpt_final[:, : ynpt.shape[1], :] = ynpt
                ynpt = ynpt_final

        def do_pad_all(tensors):
            for t in tensors:
                for t_i in t:
                    t_i[-1] = torch.zeros_like(t_i[-1], device=device, dtype=dtype)

        # PyTorch implementation returns non-zero junk in the padding
        # locations; overwrite it so that the comparison works out.
        if use_padding:
            ypt[0][-1] = torch.zeros_like(ypt[0][-1], device=device, dtype=dtype)
            ynpt[0][-1] = torch.zeros_like(ynpt[0][-1], device=device, dtype=dtype)
            if pad_all:
                do_pad_all((ypt, ynpt))
            # Zero the last row of each TxT weight matrix
            if need_weights:
                if average_attn_weights:
                    weight_pt[0][-1] = torch.zeros_like(
                        weight_pt[0][-1], device=device, dtype=dtype
                    )
                    weight_npt[0][-1] = torch.zeros_like(
                        weight_npt[0][-1], device=device, dtype=dtype
                    )
                    if pad_all:
                        do_pad_all((weight_pt, weight_npt))
                else:
                    for nh in range(num_heads):
                        weight_pt[0][nh][-1] = torch.zeros_like(
                            weight_pt[0][nh][-1], device=device, dtype=dtype
                        )
                        weight_npt[0][nh][-1] = torch.zeros_like(
                            weight_npt[0][nh][-1], device=device, dtype=dtype
                        )

        if dtype == torch.half:
            torch.testing.assert_close(
                ypt, ynpt.to(torch.float32), atol=5e-2, rtol=2e-3
            )
        else:
            # High rtol seems necessary for
            # test_native_multihead_attention_cpu_float32 on Windows,
            # otherwise 2e-4 would likely be fine.
            torch.testing.assert_close(ypt, ynpt, atol=5e-2, rtol=2e-3)

        if need_weights:
            torch.testing.assert_close(
                weight_pt, weight_npt.to(torch.float32), atol=5e-2, rtol=2e-3
            )
        else:
            assert weight_pt is None
            assert weight_npt is None

    @parametrize("dtype", [torch.float, torch.half])
    @parametrize("use_nt", [False])
    @parametrize("use_padding, pad_all", [(False, False)])
    @parametrize("need_weights", [False])
    @parametrize("average_attn_weights", [False, True])
    @torch.no_grad()
    def test_native_multihead_self_attention(
        self, dtype, use_nt, need_weights, average_attn_weights, use_padding, pad_all
    ):
        for need_weights in (False, not pad_all):
            self._test_multihead_attention_impl(
                dtype,
                "self",
                use_nt=use_nt,
                use_padding=use_padding,
                pad_all=pad_all,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
            )

    @parametrize("dtype", [torch.float, torch.half])
    @torch.no_grad()
    def test_native_multihead_encoder_decoder_attention(self, dtype):
        self._test_multihead_attention_impl(
            dtype,
            "encdec",
            use_nt=False,
            need_weights=False,
            average_attn_weights=False,
        )

    @parametrize("dtype", [torch.float, torch.half])
    @torch.no_grad()
    def test_native_multihead_attention(self, dtype):
        self._test_multihead_attention_impl(
            dtype,
            "generic",
            use_nt=False,
            need_weights=False,
            average_attn_weights=False,
        )
