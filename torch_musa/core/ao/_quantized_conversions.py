"""Quantized weight conversion utilities for MUSA mixed-dtypes linear operations.

current kernel only supports int4 weight and 16 input dtype.

Typical usage:
    from torch_musa.core.ao._quantized_conversions import (
        prepack_int4_weight_for_mixed_dtypes_linear,
        prepack_scale_for_mixed_dtypes_linear,
    )

    # Offline weight preparation
    weight_packed = prepack_int4_weight_for_mixed_dtypes_linear(weight_int4)
    scale_packed  = prepack_scale_for_mixed_dtypes_linear(scale, K)

    # Inference
    output = torch.ops.aten._mixed_dtypes_linear(input, weight_packed, scale_packed)
"""

import torch


def prepack_int4_weight_for_mixed_dtypes_linear(
    weight: torch.Tensor,
    block_n: int = 256,
    block_k: int = 128,
) -> torch.Tensor:
    """Reorder and pack int4 weight into the tiled layout for MUTLASS W4A16 GEMM.

    The MUTLASS mainloop loads weight tiles via TME with shape
    (BlockN, BlockKBytes, n_tiles, k_byte_tiles).  This function converts a
    standard (K, N) int4 weight matrix into that layout.

    Each output byte packs two int4 values along the K dimension:
    lower nibble = weight[k_even], upper nibble = weight[k_odd].

    Args:
        weight: (K, N) tensor with int4 values stored in int8/uint8.
            Values must be in the range [-8, 7] (signed int4) or [0, 15] (unsigned int4).
        block_n: Tile N dimension (must match kernel policy, default 256).
        block_k: Tile K dimension (must match kernel policy, default 128).

    Returns:
        Packed uint8 tensor of shape (n_tiles, k_tiles, block_n, block_k // 2),
        laid out contiguously for TME loading.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D (K, N), got {weight.dim()}-D")
    if weight.dtype not in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"weight must have integer dtype, got {weight.dtype}")
    if block_k % 2 != 0:
        raise ValueError(f"block_k must be even, got {block_k}")

    k, n = weight.shape
    k_bytes = block_k // 2

    q_u4 = (weight.to(torch.int16) & 0xF).to(torch.uint8)

    k_padded = ((k + block_k - 1) // block_k) * block_k
    n_padded = ((n + block_n - 1) // block_n) * block_n

    if k_padded != k or n_padded != n:
        q_pad = torch.zeros(k_padded, n_padded, dtype=torch.uint8, device=weight.device)
        q_pad[:k, :n] = q_u4
        q_u4 = q_pad

    k_tiles = k_padded // block_k
    n_tiles = n_padded // block_n

    # Reshape into tiles: (k_tiles, block_k, n_tiles, block_n)
    q_tiled = q_u4.reshape(k_tiles, block_k, n_tiles, block_n)

    # Permute to target layout: (n_tiles, k_tiles, block_n, block_k)
    q_tiled = q_tiled.permute(2, 0, 3, 1).contiguous()

    # Pack pairs along K into uint8: low nibble + high nibble
    low_nibble = q_tiled[..., 0::2]
    high_nibble = q_tiled[..., 1::2]
    packed = (low_nibble | (high_nibble << 4)).contiguous()

    assert packed.shape == (n_tiles, k_tiles, block_n, k_bytes)
    return packed.view(k_padded, n_padded // 2)


def prepack_scale_for_mixed_dtypes_linear(
    scale: torch.Tensor,
    k: int,
    block_k: int = 128,
    block_n: int = 256,
) -> torch.Tensor:
    """Prepare scale for MUTLASS W4A16 GEMM.

    Current mixed_dtypes_linear kernel requires scale_k = k_tiles
    (one scale per K-tile per N-column).

    Args:
        scale: Per-channel (N,) or per-group (scale_k, N), any float dtype.
        k: The K dimension of the weight matrix.
        block_k: Kernel tile K (default 128), should be same in mixed_dtypes_linear dispatch policy.
        block_n: Kernel tile N (default 256), should be same in mixed_dtypes_linear dispatch policy.

    Returns:
        Float32 tensor of shape (k_tiles, N_padded).
    """
    if scale.dim() == 1:
        scale_2d = scale.unsqueeze(0)  # (1, N)
    elif scale.dim() == 2:
        scale_2d = scale
    else:
        raise ValueError(f"scale must be 1-D or 2-D, got {scale.dim()}-D")

    orig_scale_k, n = scale_2d.shape
    if k % orig_scale_k != 0:
        raise ValueError(f"K ({k}) must be divisible by scale_k ({orig_scale_k})")

    group_size = k // orig_scale_k
    if group_size < block_k:
        raise ValueError(
            f"group_size ({group_size}) must be >= block_k ({block_k}). "
            f"Sub-tile group sizes are not yet supported."
        )
    if group_size % block_k != 0:
        raise ValueError(
            f"group_size ({group_size}) must be a multiple of block_k ({block_k})"
        )

    k_tiles = (k + block_k - 1) // block_k

    # Expand: repeat each row (group_size // block_k) times → (k_tiles, N)
    if orig_scale_k != k_tiles:
        repeat_factor = group_size // block_k
        scale_2d = scale_2d.repeat_interleave(repeat_factor, dim=0)

    # Pad N to block_n alignment for TME read safety
    n_padded = ((n + block_n - 1) // block_n) * block_n
    if n_padded != n:
        pad = torch.zeros(
            k_tiles, n_padded, dtype=scale_2d.dtype, device=scale_2d.device
        )
        pad[:, :n] = scale_2d
        scale_2d = pad

    # float32 + contiguous
    if scale_2d.dtype != torch.float32:
        scale_2d = scale_2d.to(torch.float32)
    if not scale_2d.is_contiguous():
        scale_2d = scale_2d.contiguous()

    return scale_2d
