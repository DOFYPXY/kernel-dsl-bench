#!/usr/bin/env python3
"""
Benchmark Conv2D (NCHW), currently specialized to:
- N=1
- K=3x3
- stride=1
- padding=1

Compare:
- PyTorch baseline (torch.nn.functional.conv2d)
- Triton custom kernel
"""

import argparse
import sys
import torch

sys.path.insert(0, "..")

from common import print_gpu_info, benchmark, verify_correctness, get_dtype, add_common_args
from conv2d_torch import torch_conv2d
from conv2d_triton import triton_conv2d_3x3_s1p1
from conv2d_tk import tk_conv2d


def main():
    parser = argparse.ArgumentParser(description="Benchmark Conv2D (3x3, s=1, p=1)")

    add_common_args(parser)

    # Conv2D-specific args
    parser.add_argument("--n", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--cin", type=int, default=64, help="Input channels (default: 64)")
    parser.add_argument("--cout", type=int, default=64, help="Output channels (default: 64)")
    parser.add_argument("--h", type=int, default=56, help="Input height (default: 56)")
    parser.add_argument("--w", type=int, default=56, help="Input width (default: 56)")

    args = parser.parse_args()

    if args.n != 1:
        print("WARNING: Triton kernel currently supports N=1 only; forcing n=1")
        args.n = 1

    print_gpu_info()
    print("\nBenchmark: Conv2D (3x3, stride=1, padding=1)")
    print()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dtype = get_dtype(args.dtype)

    x = torch.randn((args.n, args.cin, args.h, args.w), device="cuda", dtype=dtype)
    w = torch.randn((args.cout, args.cin, 3, 3), device="cuda", dtype=dtype)

    print("Configuration:")
    print(f"  Implementation: {args.impl}")
    print(f"  Shape: x={tuple(x.shape)}, w={tuple(w.shape)}")
    print(f"  Data type: {args.dtype}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Timed iterations: {args.iters}")
    print()

    if args.impl == "torch":
        fn = lambda _x, _w: torch_conv2d(_x, _w, bias=None, stride=1, padding=1)
    elif args.impl == "triton":
        fn = triton_conv2d_3x3_s1p1
    elif args.impl == "tk":
        fn = lambda _x, _w: tk_conv2d(_x, _w, bias=None, stride=1, padding=1)
    else:
        print("JAX impl not provided for conv2d in this project yet.", file=sys.stderr)
        sys.exit(1)

    print("Running benchmark...")
    mean_ms, stddev_ms = benchmark(fn, x, w, warmup_iters=args.warmup, timed_iters=args.iters)

    print(f"Results ({args.impl}):")
    print(f"  Mean time: {mean_ms:.4f} ms")
    print(f"  Stddev: {stddev_ms:.4f} ms")

    if args.impl == "triton":
        print()
        print("Verifying correctness...")
        torch_out = torch_conv2d(x, w, bias=None, stride=1, padding=1).to(torch.float32)
        triton_out = triton_conv2d_3x3_s1p1(x, w).to(torch.float32)

        is_correct, max_abs_diff = verify_correctness(triton_out, torch_out, atol=1e-3, rtol=1e-3)
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        if not is_correct:
            print("WARNING: Numerical difference detected!", file=sys.stderr)
            sys.exit(1)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()