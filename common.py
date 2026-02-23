"""Shared utilities for GPU kernel benchmarking."""

import argparse
import sys
from typing import Tuple

import torch
import triton


def print_gpu_info():
    """Print GPU and library version information."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Triton version: {triton.__version__}")
    print("=" * 80)


def benchmark(
    fn,
    *args,
    warmup_iters: int = 20,
    timed_iters: int = 200,
    **kwargs,
) -> Tuple[float, float]:
    """
    Benchmark a kernel function with CUDA event timing.
    
    Args:
        fn: Function to benchmark
        *args: Positional arguments to fn
        warmup_iters: Number of warmup iterations
        timed_iters: Number of timed iterations
        **kwargs: Keyword arguments to fn
    
    Returns:
        (mean_time_ms, stddev_time_ms)
    """
    # Warmup iterations
    for _ in range(warmup_iters):
        fn(*args, **kwargs)
    
    torch.cuda.synchronize()
    
    # Timed iterations
    times = []
    for _ in range(timed_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    import statistics
    mean_ms = statistics.mean(times)
    stddev_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    
    return mean_ms, stddev_ms


def verify_correctness(
    result: torch.Tensor,
    baseline: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Tuple[bool, float]:
    """
    Verify result matches baseline.
    
    Args:
        result: Result tensor to verify
        baseline: Baseline reference tensor
        atol: Absolute tolerance
        rtol: Relative tolerance
    
    Returns:
        (is_correct, max_abs_diff)
    """
    max_abs_diff = (result - baseline).abs().max().item()
    is_close = torch.allclose(result, baseline, atol=atol, rtol=rtol)
    
    return is_close, max_abs_diff


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return dtype_map[dtype_str]

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common benchmark arguments to an argument parser.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--impl",
        choices=["torch", "triton", "jax"],
        required=True,
        help="Which implementation to benchmark",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp32",
        help="Data type (default: fp32)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations (default: 20)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Timed iterations (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )