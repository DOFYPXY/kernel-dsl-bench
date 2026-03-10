"""Shared utilities for GPU kernel benchmarking."""

import argparse
import os
import site
import stat
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
    atol: float = 1,
    rtol: float = 1,
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
        choices=["torch", "triton", "jax", "tk", "tilelang"],
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
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--verify-device",
        type=int,
        default=None,
        help="GPU device ID for verification (default: same as --device)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear GPU cache between operations",
    )


# ---------------------------------------------------------------------------
# Subprocess environment helpers
# (used by benchmark_all.py and size_sweep scripts to enable TileLang JIT)
# ---------------------------------------------------------------------------

def _find_real_nvcc() -> str:
    """Find the real nvcc binary, skipping any wrapper we may have created."""
    wrapper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".nvcc_wrapper", "bin")
    home = os.path.expanduser("~")
    candidates = [
        "/usr/local/cuda",
        os.path.join(home, ".conda", "envs", "tk_env"),
        os.path.join(home, ".conda", "envs", "base"),
    ]
    for path_dir in os.environ.get("PATH", "").split(":"):
        if path_dir == wrapper_dir:
            continue
        nvcc_bin = os.path.join(path_dir, "nvcc")
        if os.path.isfile(nvcc_bin):
            return nvcc_bin
    for root in candidates:
        nvcc_bin = os.path.join(root, "bin", "nvcc")
        if os.path.isfile(nvcc_bin):
            return nvcc_bin
    return ""


def _ensure_nvcc_wrapper(real_nvcc: str, extra_include_dirs: list) -> str:
    """Create a thin nvcc wrapper that injects -I flags for CUDA 12.x headers.

    TileLang calls ``nvcc`` by name and does not pass extra include dirs.
    The conda nvcc 12.4 ships an older ``cuda.h`` lacking TMA types needed by
    CUTLASS sm90 headers.  The wrapper prepends the venv's CUDA 12.8 headers
    so they are found before the compiler's own system headers.

    Returns the directory containing the wrapper (prepend to PATH).
    """
    wrapper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".nvcc_wrapper", "bin")
    os.makedirs(wrapper_dir, exist_ok=True)
    wrapper_path = os.path.join(wrapper_dir, "nvcc")
    include_flags = " ".join(f'-I"{d}"' for d in extra_include_dirs)
    script = f"#!/bin/sh\nexec \"{real_nvcc}\" {include_flags} \"$@\"\n"
    with open(wrapper_path, "w") as f:
        f.write(script)
    os.chmod(wrapper_path, os.stat(wrapper_path).st_mode
             | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper_dir


def build_subprocess_env() -> dict:
    """Return an env dict suitable for subprocess calls that run TileLang kernels.

    Fixes three issues that arise when running under the project venv:

    1. ``libnvrtc.so.12`` is inside venv's nvidia package tree but not on
       ``LD_LIBRARY_PATH`` — TVM's .so cannot load it.
    2. No ``nvcc`` binary in the venv — tilelang's ``find_cuda_path()`` fails.
    3. The conda nvcc 12.4 ``cuda.h`` lacks TMA types required by CUTLASS sm90
       headers bundled with tilelang — compilation fails even for sm_75.
    """
    env = os.environ.copy()

    nvidia_lib_dirs = []
    nvidia_include_dirs = []
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_dir):
            for pkg in sorted(os.listdir(nvidia_dir)):
                lib_path = os.path.join(nvidia_dir, pkg, "lib")
                if os.path.isdir(lib_path):
                    nvidia_lib_dirs.append(lib_path)
                inc_path = os.path.join(nvidia_dir, pkg, "include")
                if os.path.isdir(inc_path):
                    nvidia_include_dirs.append(inc_path)
            break

    if nvidia_lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        extra = ":".join(nvidia_lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{extra}:{existing}" if existing else extra

    real_nvcc = _find_real_nvcc()
    if real_nvcc:
        cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(real_nvcc)))
        env.setdefault("CUDA_PATH", cuda_root)
        if nvidia_include_dirs:
            wrapper_dir = _ensure_nvcc_wrapper(real_nvcc, nvidia_include_dirs)
            current_path = env.get("PATH", "")
            if wrapper_dir not in current_path.split(":"):
                env["PATH"] = f"{wrapper_dir}:{current_path}"
        else:
            cuda_bin = os.path.join(cuda_root, "bin")
            current_path = env.get("PATH", "")
            if cuda_bin not in current_path.split(":"):
                env["PATH"] = f"{cuda_bin}:{current_path}"

    return env