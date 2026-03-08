#!/usr/bin/env python3
"""
Benchmark 2D Convolution kernel: y = conv2d(x, weight)

This module provides a benchmarking harness for comparing 2D convolution
implementations across different GPU programming frameworks:
- PyTorch (baseline, uses cuDNN)
- Triton  (direct outer-product accumulation)
- ThunderKittens (direct tiled convolution, sm_75)
"""

import argparse
import sys

import torch

# Add parent directory to path to import common utilities
sys.path.insert(0, "..")

from common import (
    print_gpu_info,
    benchmark,
    verify_correctness,
    get_dtype,
    add_common_args,
)
from conv2d_torch  import torch_conv2d
from conv2d_triton import triton_conv2d
from conv2d_tk     import tk_conv2d


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 2D Convolution: y = conv2d(x, weight)"
    )

    add_common_args(parser)

    # Conv2d-specific arguments
    parser.add_argument("--batch",   type=int, default=8,   help="Batch size (default: 8)")
    parser.add_argument("--c_in",    type=int, default=32,  help="Input channels (default: 32)")
    parser.add_argument("--c_out",   type=int, default=64,  help="Output channels (default: 64)")
    parser.add_argument("--height",  type=int, default=56,  help="Input height (default: 56)")
    parser.add_argument("--width",   type=int, default=56,  help="Input width (default: 56)")
    parser.add_argument("--kh",      type=int, default=3,   help="Kernel height (default: 3)")
    parser.add_argument("--kw",      type=int, default=3,   help="Kernel width (default: 3)")
    parser.add_argument("--stride",  type=int, default=1,   help="Stride (default: 1)")
    parser.add_argument("--padding", type=int, default=1,   help="Padding (default: 1)")

    args = parser.parse_args()

    # TK conv2d kernel requires fp32
    if args.impl == "tk" and args.dtype != "fp32":
        print(f"Note: TK Conv2d kernel requires fp32; overriding dtype from {args.dtype} to fp32.")
        args.dtype = "fp32"

    print_gpu_info()
    print(f"\nBenchmark: 2D Convolution (y = conv2d(x, weight))")
    print()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dtype = get_dtype(args.dtype)

    N, C_in, H, W  = args.batch, args.c_in,  args.height, args.width
    C_out, KH, KW  = args.c_out, args.kh,    args.kw
    stride          = args.stride
    padding         = args.padding

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1

    x      = torch.randn(N, C_in,  H, W,   dtype=dtype, device="cuda")
    weight = torch.randn(C_out, C_in, KH, KW, dtype=dtype, device="cuda")

    # FLOP count: 2 * N * C_out * OH * OW * C_in * KH * KW
    flops = 2 * N * C_out * OH * OW * C_in * KH * KW

    print(f"Configuration:")
    print(f"  Implementation: {args.impl}")
    print(f"  Input shape:  ({N}, {C_in}, {H}, {W})")
    print(f"  Kernel shape: ({C_out}, {C_in}, {KH}, {KW})")
    print(f"  Output shape: ({N}, {C_out}, {OH}, {OW})")
    print(f"  Stride:   {stride}")
    print(f"  Padding:  {padding}")
    print(f"  Data type: {args.dtype}")
    print(f"  FLOPs per conv2d: {flops:,}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Timed iterations: {args.iters}")
    print()

    # Select implementation
    if args.impl == "torch":
        fn = lambda: torch_conv2d(x, weight, stride=stride, padding=padding)
    elif args.impl == "triton":
        fn = lambda: triton_conv2d(x, weight, stride=stride, padding=padding)
    else:  # tk
        fn = lambda: tk_conv2d(x, weight, stride=stride, padding=padding)

    print("Running benchmark...")
    mean_ms, stddev_ms = benchmark(
        fn, warmup_iters=args.warmup, timed_iters=args.iters
    )

    tflops = (flops / (mean_ms * 1e-3)) / 1e12

    print(f"Results ({args.impl}):")
    print(f"  Mean time:   {mean_ms:.4f} ms")
    print(f"  Stddev:      {stddev_ms:.4f} ms")
    print(f"  Performance: {tflops:.2f} TFLOPS")

    # Verify correctness against PyTorch
    if args.impl in ["triton", "tk"]:
        print()
        print("Verifying correctness...")
        torch_result = torch_conv2d(x, weight, stride=stride, padding=padding)

        if args.impl == "triton":
            impl_result = triton_conv2d(x, weight, stride=stride, padding=padding)
        else:  # tk
            impl_result = tk_conv2d(x, weight, stride=stride, padding=padding)

        # Use slightly relaxed tolerance for float arithmetic differences
        is_correct, max_abs_diff = verify_correctness(
            impl_result, torch_result, atol=1e-4, rtol=1e-4
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
