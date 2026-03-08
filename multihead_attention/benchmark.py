#!/usr/bin/env python3
"""
Benchmark Scaled Dot-Product Multihead Attention:
  Out = softmax(Q * K^T / sqrt(D)) * V

Benchmarking harness comparing:
- PyTorch (baseline, uses FlashAttention v2 if available)
- Triton  (Flash Attention forward pass)
- ThunderKittens (Flash Attention forward pass, sm_75)
"""

import argparse
import math
import sys

import torch

sys.path.insert(0, "..")

from common import (
    print_gpu_info,
    benchmark,
    verify_correctness,
    get_dtype,
    add_common_args,
)
from multihead_attention_torch   import torch_multihead_attention
from multihead_attention_triton  import triton_multihead_attention
from multihead_attention_tk      import tk_multihead_attention


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Multihead Attention: Out = softmax(QK^T/sqrt(D)) * V"
    )

    add_common_args(parser)

    parser.add_argument("--batch",   type=int, default=4,   help="Batch size  (default: 4)")
    parser.add_argument("--heads",   type=int, default=8,   help="Num heads   (default: 8)")
    parser.add_argument("--seqlen",  type=int, default=512, help="Seq length  (default: 512)")
    parser.add_argument("--headdim", type=int, default=64,  help="Head dim    (default: 64)")

    args = parser.parse_args()

    # TK MHA requires fp32 and D ≤ 64
    if args.impl == "tk":
        if args.dtype != "fp32":
            print(f"Note: TK MHA requires fp32; overriding dtype from {args.dtype} to fp32.")
            args.dtype = "fp32"
        if args.headdim > 64:
            print(f"Note: TK MHA requires head_dim ≤ 64; clamping from {args.headdim} to 64.")
            args.headdim = 64

    print_gpu_info()
    print(f"\nBenchmark: Multihead Attention (Out = softmax(QK^T/sqrt(D)) * V)")
    print()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dtype = get_dtype(args.dtype)
    B, H, S, D = args.batch, args.heads, args.seqlen, args.headdim
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, S, D, dtype=dtype, device="cuda")

    # FLOPs: 2 × B × H × (2 × S² × D)  =  4 × B × H × S² × D
    flops = 4 * B * H * S * S * D

    print(f"Configuration:")
    print(f"  Implementation: {args.impl}")
    print(f"  Shape: B={B}, H={H}, S={S}, D={D}")
    print(f"  Data type: {args.dtype}")
    print(f"  FLOPs per forward: {flops:,}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Timed iterations: {args.iters}")
    print()

    if args.impl == "torch":
        fn = lambda: torch_multihead_attention(q, k, v, scale=scale)
    elif args.impl == "triton":
        fn = lambda: triton_multihead_attention(q, k, v, scale=scale)
    else:  # tk
        fn = lambda: tk_multihead_attention(q, k, v, scale=scale)

    print("Running benchmark...")
    mean_ms, stddev_ms = benchmark(
        fn, warmup_iters=args.warmup, timed_iters=args.iters
    )

    tflops = (flops / (mean_ms * 1e-3)) / 1e12

    print(f"Results ({args.impl}):")
    print(f"  Mean time:   {mean_ms:.4f} ms")
    print(f"  Stddev:      {stddev_ms:.4f} ms")
    print(f"  Performance: {tflops:.2f} TFLOPS")

    if args.impl in ["triton", "tk"]:
        print()
        print("Verifying correctness...")
        torch_result = torch_multihead_attention(q, k, v, scale=scale)

        if args.impl == "triton":
            impl_result = triton_multihead_attention(q, k, v, scale=scale)
        else:  # tk
            impl_result = tk_multihead_attention(q, k, v, scale=scale)

        is_correct, max_abs_diff = verify_correctness(
            impl_result, torch_result, atol=1e-3, rtol=1e-3
        )

        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")

        if not is_correct:
            print("WARNING: Numerical difference detected!", file=sys.stderr)
            sys.exit(1)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
