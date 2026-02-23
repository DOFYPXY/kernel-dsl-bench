#!/usr/bin/env python3
"""
Benchmark Fused Multiply-Add (FMA) kernel: y = x * a + b

This module provides a benchmarking harness for comparing FMA implementations
across different GPU programming DSLs:
- PyTorch (baseline)
- Triton
- JAX
"""

import argparse
import sys

import torch

# Add parent directory to path to import common utilities
sys.path.insert(0, '..')

from common import print_gpu_info, benchmark, verify_correctness, get_dtype, add_common_args
from fma_torch import torch_fma
from fma_triton import triton_fma
from fma_jax import jax_fma


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Fused Multiply-Add (FMA): y = x * a + b"
    )
    
    # Add common arguments
    add_common_args(parser)
    
    # Add FMA-specific arguments
    parser.add_argument(
        "--n",
        type=int,
        default=10_000_000,
        help="Number of elements (default: 10M)",
    )
    
    args = parser.parse_args()
    
    print_gpu_info()
    print(f"\nBenchmark: Fused Multiply-Add (y = x * a + b)")
    print()
    
    # Setup random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Determine data type
    dtype = get_dtype(args.dtype)
    
    # Allocate test data
    x = torch.randn(args.n, dtype=dtype, device="cuda")
    a = 2.5
    b = 1.3
    
    print(f"Configuration:")
    print(f"  Implementation: {args.impl}")
    print(f"  Elements: {args.n:,}")
    print(f"  Data type: {args.dtype}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Timed iterations: {args.iters}")
    print()
    
    # Select implementation
    if args.impl == "torch":
        fn = torch_fma
    elif args.impl == "triton":
        fn = triton_fma
    else:  # jax
        fn = jax_fma
    
    # Run benchmark
    print("Running benchmark...")
    mean_ms, stddev_ms = benchmark(
        fn, x, a, b, warmup_iters=args.warmup, timed_iters=args.iters
    )
    
    print(f"Results ({args.impl}):")
    print(f"  Mean time: {mean_ms:.4f} ms")
    print(f"  Stddev: {stddev_ms:.4f} ms")
    
    # Verify correctness
    if args.impl in ["triton", "jax"]:
        print()
        print("Verifying correctness...")
        torch_result = torch_fma(x, a, b)
        
        if args.impl == "triton":
            impl_result = triton_fma(x, a, b)
        else:  # jax
            impl_result = jax_fma(x, a, b)
        
        is_correct, max_abs_diff = verify_correctness(impl_result, torch_result)
        
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        
        if not is_correct:
            print("WARNING: Numerical difference detected!", file=sys.stderr)
            sys.exit(1)
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
