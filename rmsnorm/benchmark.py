#!/usr/bin/env python3
"""
Benchmark RMSNorm kernel.

This module provides a benchmarking harness for comparing RMSNorm
implementations across different GPU programming DSLs:
- PyTorch (baseline)
- Triton
"""

import argparse
import sys

import torch

# Add parent directory to path to import common utilities
sys.path.insert(0, '..')

from common import print_gpu_info, benchmark, verify_correctness, get_dtype, add_common_args
from rmsnorm_torch import torch_rmsnorm
from rmsnorm_triton import triton_rmsnorm


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RMSNorm"
    )

    # Add common arguments
    add_common_args(parser)

    # Add RMSNorm-specific arguments
    parser.add_argument(
        "--batch",
        type=int,
        default=4096,
        help="Batch size (default: 4096)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=1024,
        help="Hidden dimension (default: 1024)",
    )

    args = parser.parse_args()

    print_gpu_info()
    print(f"\nBenchmark: RMSNorm")
    print()

    # Setup random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Determine data type
    dtype = get_dtype(args.dtype)

    # Allocate test data
    x = torch.randn(args.batch, args.hidden, dtype=dtype, device="cuda")
    weight = torch.ones(args.hidden, dtype=dtype, device="cuda")

    print(f"Configuration:")
    print(f"  Implementation: {args.impl}")
    print(f"  Shape: ({args.batch}, {args.hidden})")
    print(f"  Data type: {args.dtype}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Timed iterations: {args.iters}")
    print()

    # Select implementation
    if args.impl == "torch":
        fn = torch_rmsnorm
    elif args.impl == "triton":
        fn = triton_rmsnorm
    else:
        print("JAX not implemented for RMSNorm", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    print("Running benchmark...")
    mean_ms, stddev_ms = benchmark(
        fn,
        x,
        weight,
        warmup_iters=args.warmup,
        timed_iters=args.iters,
    )

    print(f"Results ({args.impl}):")
    print(f"  Mean time: {mean_ms:.4f} ms")
    print(f"  Stddev: {stddev_ms:.4f} ms")

    # Verify correctness
    if args.impl == "triton":
        print()
        print("Verifying correctness...")
        torch_result = torch_rmsnorm(x, weight)
        triton_result = triton_rmsnorm(x, weight)

        is_correct, max_abs_diff = verify_correctness(triton_result, torch_result)

        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")

        if not is_correct:
            print("WARNING: Numerical difference detected!", file=sys.stderr)
            sys.exit(1)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()