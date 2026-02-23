#!/usr/bin/env python3
"""
Benchmark Matrix Multiplication (MatMul) kernel: C = A @ B

This module provides a benchmarking harness for comparing MatMul implementations
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
from matmul_torch import torch_matmul
from matmul_triton import triton_matmul
from matmul_jax import jax_matmul


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Matrix Multiplication (MatMul): C = A @ B"
    )
    
    # Add common arguments
    add_common_args(parser)
    
    # Override dtype default for MatMul
    parser.set_defaults(dtype="fp16")
    
    # Add MatMul-specific arguments
    parser.add_argument(
        "--m",
        type=int,
        default=1024,
        help="Matrix dimension M (rows of A, rows of C) (default: 1024)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1024,
        help="Matrix dimension N (columns of B, columns of C) (default: 1024)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1024,
        help="Matrix dimension K (columns of A, rows of B) (default: 1024)",
    )
    
    args = parser.parse_args()
    
    print_gpu_info()
    print(f"\nBenchmark: Matrix Multiplication (C = A @ B)")
    print()
    
    # Setup random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Determine data type
    dtype = get_dtype(args.dtype)
    
    # Allocate test matrices
    a = torch.randn((args.m, args.k), dtype=dtype, device="cuda")
    b = torch.randn((args.k, args.n), dtype=dtype, device="cuda")
    
    M, K = a.shape
    K, N = b.shape
    
    # Calculate FLOPS for performance metric
    # Each output element: K multiplies + K-1 adds ≈ 2K operations
    flops = 2 * M * N * K
    
    print(f"Configuration:")
    print(f"  Implementation: {args.impl}")
    print(f"  Matrix A shape: ({M}, {K})")
    print(f"  Matrix B shape: ({K}, {N})")
    print(f"  Matrix C shape: ({M}, {N})")
    print(f"  Data type: {args.dtype}")
    print(f"  FLOPs per matmul: {flops:,}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Timed iterations: {args.iters}")
    print()
    
    # Select implementation
    if args.impl == "torch":
        fn = torch_matmul
    elif args.impl == "triton":
        fn = triton_matmul
    else:  # jax
        fn = jax_matmul
    
    # Run benchmark
    print("Running benchmark...")
    mean_ms, stddev_ms = benchmark(
        fn, a, b, warmup_iters=args.warmup, timed_iters=args.iters
    )
    
    # Calculate TFLOPS
    tflops = (flops / (mean_ms * 1e-3)) / 1e12
    
    print(f"Results ({args.impl}):")
    print(f"  Mean time: {mean_ms:.4f} ms")
    print(f"  Stddev: {stddev_ms:.4f} ms")
    print(f"  Performance: {tflops:.2f} TFLOPS")
    
    # Verify correctness
    if args.impl in ["triton", "jax"]:
        print()
        print("Verifying correctness...")
        torch_result = torch_matmul(a, b)
        
        if args.impl == "triton":
            impl_result = triton_matmul(a, b)
        else:  # jax
            impl_result = jax_matmul(a, b)
        
        # Use relaxed tolerances for matmul due to accumulation errors
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        
        is_correct, max_abs_diff = verify_correctness(
            impl_result, torch_result, atol=atol, rtol=rtol
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
