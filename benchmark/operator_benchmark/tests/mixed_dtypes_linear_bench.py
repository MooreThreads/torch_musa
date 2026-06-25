#!/usr/bin/env python3
"""Benchmark for _mixed_dtypes_linear (W4A16 quantized GEMM) — TFlops measurement."""

import argparse
import statistics

import torch

from torch_musa.core.ao._quantized_conversions import (
    prepack_int4_weight_for_mixed_dtypes_linear,
    prepack_scale_for_mixed_dtypes_linear,
)


def _quantize_s4_per_group(w, group_size):
    k, n = w.shape
    scale_k = (k + group_size - 1) // group_size
    w_q = torch.empty((k, n), dtype=torch.int16, device=w.device)
    scales = torch.empty((scale_k, n), dtype=torch.float32, device=w.device)
    for gi in range(scale_k):
        k0 = gi * group_size
        k1 = min((gi + 1) * group_size, k)
        w_slice = w[k0:k1].to(torch.float32)
        scale = w_slice.abs().amax(dim=0).clamp(min=1e-6) / 7.0
        q = torch.round(w_slice / scale).clamp(-8, 7).to(torch.int16)
        w_q[k0:k1] = q
        scales[gi] = scale
    return w_q, scales


def _prepare_data(m, k, n, group_size, dtype, device="musa"):
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=dtype, device=device)
    w = torch.randn(k, n, dtype=dtype, device=device)
    w_q, scales = _quantize_s4_per_group(w, group_size)
    w_packed = prepack_int4_weight_for_mixed_dtypes_linear(w_q)
    s_packed = prepack_scale_for_mixed_dtypes_linear(scales, k)
    return a, w_packed, s_packed


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_one(m, n, k, group_size, dtype, warmup, repeat):
    a, w_packed, s_packed = _prepare_data(m, k, n, group_size, dtype)

    for _ in range(warmup):
        torch.ops.aten._mixed_dtypes_linear(a, w_packed, s_packed)
    torch.musa.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        torch.ops.aten._mixed_dtypes_linear(a, w_packed, s_packed)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))  # ms

    median_ms = statistics.median(times)
    flops = 2.0 * m * n * k
    tflops = flops / (median_ms * 1e9) if median_ms > 0 else 0.0
    return median_ms, tflops


def main():
    parser = argparse.ArgumentParser(description="W4A16 mixed-dtypes linear benchmark")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"],
                        help="Input dtype (default: fp16)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=100,
                        help="Timed iterations (default: 100)")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    group_size = args.group_size

    # constraints: N % 256 == 0, K % 128 == 0
    shapes = [
        (256,   512,   512),
        (4096,  4096,  4096),
        (32768, 4096,  4096),
        (32768, 11008, 4096),
    ]

    print(f"\nW4A16 Mixed-Dtypes Linear Benchmark ({dtype}, group_size={group_size})")
    print(f"warmup={args.warmup}  repeat={args.repeat}")
    print("-" * 60)
    print(f"{'M':>8}{'N':>8}{'K':>8}{'Time(ms)':>12}{'TFlops':>10}")
    print("-" * 60)

    for m, n, k in shapes:
        median_ms, tflops = bench_one(m, n, k, group_size, dtype,
                                      args.warmup, args.repeat)
        print(f"{m:>8}{n:>8}{k:>8}{median_ms:>12.3f}{tflops:>10.2f}")

    print("-" * 60)


if __name__ == "__main__":
    main()
