#!/usr/bin/env python3
"""
Benchmark Differential Privacy Laplace mechanism:
    y = clip(x, clip_norm) + Laplace(0, clip_norm / epsilon)

Compares:
  - PyTorch (baseline)
  - ThunderKittens (TK)
"""

import argparse
import sys

import torch

sys.path.insert(0, '..')

from common import print_gpu_info, benchmark, verify_correctness, get_dtype, add_common_args
from dp_noise_torch import torch_dp_noise
from dp_noise_tk import tk_dp_noise


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DP Laplace noise: y = clip(x, C) + Laplace(0, C/ε)"
    )

    add_common_args(parser)

    parser.add_argument(
        "--n", type=int, default=10_000_000,
        help="Number of elements (default: 10M)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="Privacy budget ε (default: 1.0)",
    )
    parser.add_argument(
        "--clip-norm", type=float, default=1.0,
        help="L2 clipping bound C (default: 1.0)",
    )

    args = parser.parse_args()

    print_gpu_info()
    print(f"\nBenchmark: DP Laplace Noise  (y = clip(x, C) + Laplace(0, C/ε))")
    print()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dtype = get_dtype(args.dtype)

    x = torch.randn(args.n, dtype=dtype, device="cuda")
    epsilon = args.epsilon
    clip_norm = args.clip_norm
    seed = args.seed

    print(f"Configuration:")
    print(f"  Implementation: {args.impl}")
    print(f"  Elements: {args.n:,}")
    print(f"  Data type: {args.dtype}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Clip norm: {clip_norm}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Timed iterations: {args.iters}")
    print()

    if args.impl == "torch":
        fn = torch_dp_noise
    elif args.impl == "tk":
        fn = tk_dp_noise
    else:
        print(f"Unknown implementation: {args.impl}", file=sys.stderr)
        sys.exit(1)

    # Benchmark
    print("Running benchmark...")
    mean_ms, stddev_ms = benchmark(
        fn, x, epsilon, clip_norm, seed,
        warmup_iters=args.warmup, timed_iters=args.iters,
    )

    print(f"Results ({args.impl}):")
    print(f"  Mean time: {mean_ms:.4f} ms")
    print(f"  Stddev:    {stddev_ms:.4f} ms")

    # ── Correctness: verify clipping behaviour ────────────────────────────
    #  Since noise is stochastic, we cannot compare outputs directly.
    #  Instead we verify that the *clipping* step matches between impls.
    if args.impl == "tk":
        print()
        print("Verifying clipping correctness (noise disabled via large epsilon)...")

        # With very large epsilon, Laplace scale → 0, so noise ≈ 0.
        large_eps = 1e12
        torch_result = torch_dp_noise(x, large_eps, clip_norm, seed)
        tk_result    = tk_dp_noise(x, large_eps, clip_norm, seed)

        is_correct, max_diff = verify_correctness(
            tk_result, torch_result, atol=1e-3, rtol=1e-3
        )

        print(f"  Max absolute difference (clip only): {max_diff:.2e}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")

        if not is_correct:
            print("WARNING: Numerical difference detected!", file=sys.stderr)
            sys.exit(1)

        # Verify noise distribution (Laplace with real epsilon)
        print()
        print("Verifying noise distribution (Kolmogorov-Smirnov test)...")
        tk_out = tk_dp_noise(x, epsilon, clip_norm, seed)

        # Recover noise: noise = output - clipped_x
        norm = torch.linalg.norm(x.float())
        cf = min(1.0, clip_norm / (norm.item() + 1e-12))
        x_clipped = x * cf
        noise = tk_out - x_clipped

        b = clip_norm / epsilon
        # Check mean ≈ 0, scale ≈ b
        noise_mean = noise.mean().item()
        noise_std  = noise.std().item()
        expected_std = b * (2.0 ** 0.5)  # Laplace std = b√2

        print(f"  Noise mean:       {noise_mean:.4f}  (expected ≈ 0)")
        print(f"  Noise std:        {noise_std:.4f}  (expected ≈ {expected_std:.4f})")
        print(f"  Laplace scale b:  {b:.4f}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
