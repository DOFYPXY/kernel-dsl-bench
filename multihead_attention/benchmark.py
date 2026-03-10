#!/usr/bin/env python3
"""
Benchmark multihead attention kernel.

This module provides a benchmarking harness for comparing multihead attention
implementations across different GPU programming DSLs.
"""

import argparse
import sys

import torch

# Add parent directory to path to import common utilities
sys.path.insert(0, "..")

from common import add_common_args, benchmark, get_dtype, print_gpu_info
from multihead_attention_torch import torch_multihead_attention
from multihead_attention_triton import triton_multihead_attention
from multihead_attention_tk import tk_multihead_attention
from multihead_attention_tilelang import tilelang_multihead_attention


def main():
    parser = argparse.ArgumentParser(description="Benchmark Multihead Attention")

    # Add common arguments
    add_common_args(parser)

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=16,
        help="Number of attention heads (default: 16)",
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=1024,
        help="Sequence length (default: 1024)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension (default: 64)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable causal masking",
    )

    args = parser.parse_args()

    print_gpu_info()
    print("\nBenchmark: Multihead Attention")
    print()

    # Setup random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Determine data type
    dtype = get_dtype(args.dtype)

    # Allocate test tensors (B, H, S, D)
    q = torch.randn(args.batch, args.heads, args.seq, args.head_dim, dtype=dtype, device="cuda")
    k = torch.randn(args.batch, args.heads, args.seq, args.head_dim, dtype=dtype, device="cuda")
    v = torch.randn(args.batch, args.heads, args.seq, args.head_dim, dtype=dtype, device="cuda")

    # Approximate FLOPs for attention: QK^T and AV, each ~2 * B * H * S * S * D
    flops = 4 * args.batch * args.heads * args.seq * args.seq * args.head_dim

    print("Configuration:")
    print(f"  Implementation: {args.impl}")
    print(f"  Shape (B, H, S, D): ({args.batch}, {args.heads}, {args.seq}, {args.head_dim})")
    print(f"  Causal: {args.causal}")
    print(f"  Data type: {args.dtype}")
    print(f"  Approx FLOPs per pass: {flops:,}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Timed iterations: {args.iters}")
    print()

    if args.impl == "torch":
        fn = torch_multihead_attention
    elif args.impl == "triton":
        fn = triton_multihead_attention
    elif args.impl == "tk":
        fn = tk_multihead_attention
    elif args.impl == "tilelang":
        fn = tilelang_multihead_attention
    else:
        print(f"{args.impl.upper()} not implemented for multihead_attention", file=sys.stderr)
        sys.exit(1)

    print("Running benchmark...")
    mean_ms, stddev_ms = benchmark(
        fn,
        q,
        k,
        v,
        warmup_iters=args.warmup,
        timed_iters=args.iters,
        causal=args.causal,
    )

    tflops = (flops / (mean_ms * 1e-3)) / 1e12

    print(f"Results ({args.impl}):")
    print(f"  Mean time: {mean_ms:.4f} ms")
    print(f"  Stddev: {stddev_ms:.4f} ms")
    print(f"  Performance: {tflops:.2f} TFLOPS")
    print()
    print("=" * 80)

    if args.impl == "triton":
        print("Verifying correctness on a smaller test case...")
        test_b = min(args.batch, 2)
        test_h = min(args.heads, 2)
        test_s = min(args.seq, 128)

        q_test = q[:test_b, :test_h, :test_s, :].contiguous()
        k_test = k[:test_b, :test_h, :test_s, :].contiguous()
        v_test = v[:test_b, :test_h, :test_s, :].contiguous()

        ref = torch_multihead_attention(q_test, k_test, v_test, causal=args.causal)
        out = triton_multihead_attention(q_test, k_test, v_test, causal=args.causal)

        max_diff = (ref - out).abs().max().item()
        print(f"Max absolute difference: {max_diff:.6e}")
        print()
        
    if args.impl == "tilelang":
        print("Verifying correctness on a smaller test case...")
        test_b = min(args.batch, 2)
        test_h = min(args.heads, 2)
        test_s = min(args.seq, 128)

        q_test = q[:test_b, :test_h, :test_s, :].contiguous()
        k_test = k[:test_b, :test_h, :test_s, :].contiguous()
        v_test = v[:test_b, :test_h, :test_s, :].contiguous()

        ref = torch_multihead_attention(q_test, k_test, v_test, causal=args.causal)
        out = tilelang_multihead_attention(q_test, k_test, v_test, causal=args.causal)

        max_diff = (ref.float() - out.float()).abs().max().item()
        print(f"Max absolute difference: {max_diff:.6e}")
        print()
    

if __name__ == "__main__":
    main()