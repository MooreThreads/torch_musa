"""Test rope forward & backward operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, redefined-builtin, unused-argument, unexpected-keyword-arg
import torch
import pytest
from torch.nn import functional as F

import torch_musa
from torch_musa import testing


def rotate_half(t: torch.Tensor) -> torch.Tensor:
    t_1, t_2 = torch.chunk(t, 2, dim=-1)
    return torch.cat((-t_2, t_1), dim=-1)


def apply_rotary_pos_emb_bshd(
    input: torch.Tensor,
    freq_cis: torch.Tensor,
    multi_latent_attention: bool = False,
    rotary_interleaved: bool = False,
    batch_first: bool = False,
):
    if freq_cis.dim() == 2:
        freq_cis = freq_cis.reshape(freq_cis.shape[0], 1, 1, freq_cis.shape[-1])

    rot_dim = freq_cis.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = input[..., :rot_dim], input[..., rot_dim:]

    if multi_latent_attention:
        t = torch.cat([t[..., 0::2], t[..., 1::2]], dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = torch.cos(freq_cis).to(t.dtype)
    sin_ = torch.sin(freq_cis).to(t.dtype)

    t = (t * cos_) + (rotate_half(t) * sin_)

    return torch.cat((t, t_pass), dim=-1)


multi_latent_attention = [False]
if torch.backends.mudnn.version() >= 3000:
    multi_latent_attention.append(True)


batch_sizes = [1, 8, 16]
dims = [256, 512, 1024]
seq_lens = [512, 1024]
dtypes = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(testing.get_musa_arch() < 22, reason="only test on arch>=22")
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("max_seq_len", seq_lens)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("multi_latent_attention", multi_latent_attention)
def test_rope(batch_size, dim, max_seq_len, dtype, multi_latent_attention):

    head_num = 32
    dim = dim // head_num
    input_shape = (max_seq_len, batch_size, head_num, dim)  # SBHD

    check_func = testing.DefaultComparator(abs_diff=1e-4, rel_diff=1e-5)

    input_ts = torch.randn(input_shape, dtype=dtype)
    freqs_cis = torch.randn(max_seq_len, dim, dtype=dtype)  # SD
    freqs_cis_4d = freqs_cis.reshape(max_seq_len, 1, 1, dim)

    cpu_ret = apply_rotary_pos_emb_bshd(input_ts, freqs_cis_4d, multi_latent_attention)

    # -----------------------------test nn.RoPE ---------------------
    input_data_out = {
        "freq_cis": freqs_cis,
        "rotary_interleaved": False,
        "batch_first": False,
        "multi_latent_attention": multi_latent_attention,
    }
    m = torch.nn.RoPE(**input_data_out).musa()
    musa_ret_out = m(input_ts.musa())
    assert check_func(cpu_ret, musa_ret_out)

    # -----------------------------test rope.out ---------------------
    if multi_latent_attention:
        input_data_out = {
            "input": input_ts.musa(),
            "freq_cis": freqs_cis.musa(),
            "rotary_interleaved": False,
            "batch_first": False,
            "multi_latent_attention": multi_latent_attention,
        }
    else:
        input_data_out = {
            "input": input_ts.musa(),
            "freq_cis": freqs_cis.musa(),
            "rotary_interleaved": False,
            "batch_first": False,
        }
    ret1 = F.rope(**input_data_out)
    assert check_func(cpu_ret, ret1)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(testing.get_musa_arch() < 22, reason="only test on arch>=22")
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("max_seq_len", seq_lens)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("multi_latent_attention", multi_latent_attention)
@pytest.mark.parametrize(
    "scenario",
    [
        "qkv_merge",
        "dist_split",
    ],
)
def test_rope_non_contiguous_scenarios(
    batch_size, dim, max_seq_len, dtype, multi_latent_attention, scenario
):

    head_num = 32
    head_dim = dim // head_num

    input_shape = (max_seq_len, batch_size, head_num, head_dim)
    input_ts = torch.randn(input_shape, dtype=dtype)

    freqs_cis = torch.randn((max_seq_len, head_dim), dtype=dtype)
    freqs_cis_4d = freqs_cis.reshape(max_seq_len, 1, 1, head_dim)
    if scenario == "qkv_merge":
        k = torch.randn(input_shape, dtype=dtype)
        v = torch.randn(input_shape, dtype=dtype)
        qkv_merged = torch.cat([input_ts, k, v], dim=-1)
        qkv_merged = qkv_merged.view(max_seq_len, batch_size, 3, head_num, head_dim)
        input_ts = qkv_merged[:, :, 0, :, :]

    elif scenario == "dist_split":
        dist_chunk = input_ts[:, :, : (head_num // 2), :]
        input_ts = dist_chunk

    cpu_ret = apply_rotary_pos_emb_bshd(input_ts, freqs_cis_4d, multi_latent_attention)

    input_ts_musa = input_ts.clone().musa()
    freqs_musa = freqs_cis.clone().musa()
    # -----------------------------test nn.RoPE ---------------------
    rope_args = {
        "freq_cis": freqs_musa,
        "rotary_interleaved": False,
        "batch_first": False,
        "multi_latent_attention": multi_latent_attention,
    }
    module = torch.nn.RoPE(**rope_args).musa()
    musa_ret = module(input_ts_musa).cpu()

    check_func = testing.DefaultComparator(abs_diff=1e-4, rel_diff=1e-5)
    assert check_func(
        cpu_ret, musa_ret
    ), f"RoPE result mismatch in scenario={scenario}, func=torch.nn.RoPE."
    # -----------------------------test rope.out ---------------------
    rope_args = {
        "input": input_ts_musa,
        "freq_cis": freqs_musa,
        "rotary_interleaved": False,
        "batch_first": False,
        "multi_latent_attention": multi_latent_attention,
    }

    musa_ret = F.rope(**rope_args).cpu()

    check_func = testing.DefaultComparator(abs_diff=1e-4, rel_diff=1e-5)
    assert check_func(
        cpu_ret, musa_ret
    ), f"RoPE result mismatch in scenario={scenario}, func=F.rope."


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(testing.get_musa_arch() < 22, reason="only test on arch>=22")
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("max_seq_len", seq_lens)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("multi_latent_attention", multi_latent_attention)
def test_rope_backward(batch_size, dim, max_seq_len, dtype, multi_latent_attention):

    head_num = 32
    dim = dim // head_num
    input_shape = (max_seq_len, batch_size, head_num, dim)  # SBHD

    input_ts = torch.randn(input_shape, dtype=dtype, requires_grad=True)
    freqs_cis = torch.randn(max_seq_len, dim, dtype=dtype)  # SD

    input_data = {
        "input": input_ts,
        "freq_cis": freqs_cis,
        "rotary_interleaved": False,
        "batch_first": False,
        "multi_latent_attention": multi_latent_attention,
    }

    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-2
    elif dtype == torch.float32:
        atol, rtol = 1e-6, 1e-6
    else:
        atol, rtol = 5e-3, 5e-2

    test = testing.OpTest(
        func=F.rope,
        refer_func=apply_rotary_pos_emb_bshd,
        input_args=input_data,
        comparators=testing.DefaultComparator(rel_diff=rtol, abs_diff=atol),
    )
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32(train=True)
    elif dtype == torch.bfloat16:
        test.check_musabf16_vs_musafp16(train=True)
    else:
        test.check_result(train=True)
