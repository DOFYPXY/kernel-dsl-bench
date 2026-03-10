#!/usr/bin/env python3
"""
Run all kernel benchmarks and collect results into a table.

This script runs all configured kernels against selected implementations
and outputs a comprehensive table with timing and performance metrics.

"""

import subprocess
import sys
import os
import re
import csv
import site
import stat
import argparse
from typing import Dict, Optional, Tuple


def _find_real_nvcc() -> str:
    """Find the real nvcc binary (excluding any wrapper we may have created)."""
    wrapper_dir = os.path.join(os.path.dirname(__file__), ".nvcc_wrapper", "bin")
    home = os.path.expanduser("~")
    # Candidate CUDA roots in priority order
    candidates = [
        "/usr/local/cuda",
        os.path.join(home, ".conda", "envs", "tk_env"),
        os.path.join(home, ".conda", "envs", "base"),
    ]
    # If nvcc is already on PATH (but not our wrapper), prefer that
    for path_dir in os.environ.get("PATH", "").split(":"):
        if path_dir == wrapper_dir:
            continue  # skip our own wrapper dir
        nvcc_bin = os.path.join(path_dir, "nvcc")
        if os.path.isfile(nvcc_bin):
            return nvcc_bin
    # Fallback: well-known CUDA roots
    for root in candidates:
        nvcc_bin = os.path.join(root, "bin", "nvcc")
        if os.path.isfile(nvcc_bin):
            return nvcc_bin
    return ""


def _ensure_nvcc_wrapper(real_nvcc: str, extra_include_dirs: list) -> str:
    """Create (or refresh) a thin nvcc wrapper script that injects -I flags.

    TileLang calls ``nvcc`` by name (without full path) and does not pass any
    extra include directories.  The installed nvcc (from a conda CUDA 12.4 env)
    ships with an older ``cuda.h`` that lacks TMA types (CUtensorMap, etc.)
    required by CUTLASS sm90 headers.  We solve this by placing a wrapper
    script earlier on PATH that delegates to the real nvcc while prepending
    ``-I <venv_cuda_include>`` so the correct (CUDA 12.8) ``cuda.h`` is found.

    Returns the directory containing the wrapper (to prepend to PATH).
    """
    wrapper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".nvcc_wrapper", "bin")
    os.makedirs(wrapper_dir, exist_ok=True)
    wrapper_path = os.path.join(wrapper_dir, "nvcc")

    include_flags = " ".join(f'-I"{d}"' for d in extra_include_dirs)
    script = (
        "#!/bin/sh\n"
        f'exec "{real_nvcc}" {include_flags} "$@"\n'
    )

    # (Re-)write the wrapper so it reflects any changes in include dirs.
    with open(wrapper_path, "w") as f:
        f.write(script)
    os.chmod(wrapper_path, os.stat(wrapper_path).st_mode
             | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return wrapper_dir


def _build_subprocess_env() -> dict:
    """Build an env dict for subprocesses with LD_LIBRARY_PATH, CUDA_PATH, and PATH.

    Fixes three issues that prevent TileLang kernels from JIT-compiling when
    running under the project venv instead of the conda ``tk_env``:

    1. ``libnvrtc.so.12`` (and other CUDA runtime libs) live inside the venv's
       nvidia package tree but are not on ``LD_LIBRARY_PATH``.  TVM's .so needs
       them at load time.

    2. The venv has no ``nvcc`` binary, so tilelang's ``find_cuda_path()`` fails.
       We locate nvcc from the conda env and export ``CUDA_PATH`` accordingly.

    3. The conda nvcc 12.4 ships a ``cuda.h`` that predates TMA APIs
       (``CUtensorMap``, ``CU_TENSOR_MAP_SWIZZLE_*``) required by CUTLASS sm90
       headers.  We create a thin wrapper ``nvcc`` script that prepends
       ``-I <cuda_runtime_include_from_venv>`` so the CUDA 12.8 headers are
       found before the compiler's own system headers.
    """
    env = os.environ.copy()

    # -- Collect nvidia lib/include dirs from the current interpreter's site-packages --
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
            break  # only need one site-packages directory

    # 1. LD_LIBRARY_PATH
    if nvidia_lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        extra = ":".join(nvidia_lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{extra}:{existing}" if existing else extra

    # 2. CUDA_PATH + PATH so tilelang can call `nvcc`
    real_nvcc = _find_real_nvcc()
    if real_nvcc:
        cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(real_nvcc)))
        env.setdefault("CUDA_PATH", cuda_root)

        # 3. nvcc wrapper that injects correct -I flags
        if nvidia_include_dirs:
            wrapper_dir = _ensure_nvcc_wrapper(real_nvcc, nvidia_include_dirs)
            current_path = env.get("PATH", "")
            if wrapper_dir not in current_path.split(":"):
                env["PATH"] = f"{wrapper_dir}:{current_path}"
        else:
            # No extra includes needed; just put the real nvcc on PATH
            cuda_bin = os.path.join(cuda_root, "bin")
            current_path = env.get("PATH", "")
            if cuda_bin not in current_path.split(":"):
                env["PATH"] = f"{cuda_bin}:{current_path}"

    return env


class BenchmarkResult:
    def __init__(self, mean_ms: Optional[float] = None, 
                 stddev_ms: Optional[float] = None, 
                 tflops: Optional[float] = None,
                 error: Optional[str] = None):
        self.mean_ms = mean_ms
        self.stddev_ms = stddev_ms
        self.tflops = tflops
        self.error = error
        
    def is_available(self):
        return self.error is None


def run_benchmark(kernel: str, impl: str, extra_args: list) -> BenchmarkResult:
    """Run a single benchmark and parse the output."""
    # Use current Python environment (assumes venv is activated)
    cmd = [sys.executable, "run.py", kernel, "--impl", impl] + extra_args
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        timeout_s = 120

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_build_subprocess_env(),
        )
        
        output = result.stdout + result.stderr
        
        # Check for "not implemented" with precise patterns only.
        # Avoid matching generic dependency warnings (e.g. "JAX not available").
        if result.returncode != 0:
            output_lc = output.lower()
            not_impl_patterns = [
                " not implemented for ",
                "jax not implemented",
                "impl not provided",
            ]
            if any(pat in output_lc for pat in not_impl_patterns):
                return BenchmarkResult(error="Not implemented")
            else:
                return BenchmarkResult(error=f"Error (exit code {result.returncode})")
        
        # Parse the output
        mean_ms = None
        stddev_ms = None
        tflops = None
        
        # Extract mean time
        mean_match = re.search(r'Mean time:\s+([\d.]+)\s+ms', output)
        if mean_match:
            mean_ms = float(mean_match.group(1))
        
        # Extract stddev
        stddev_match = re.search(r'Stddev:\s+([\d.]+)\s+ms', output)
        if stddev_match:
            stddev_ms = float(stddev_match.group(1))
        
        # Extract performance (TFLOPS) if available
        tflops_match = re.search(r'Performance:\s+([\d.]+)\s+TFLOPS', output)
        if tflops_match:
            tflops = float(tflops_match.group(1))
        
        if mean_ms is None:
            return BenchmarkResult(error="Failed to parse output")
        
        return BenchmarkResult(mean_ms=mean_ms, stddev_ms=stddev_ms, tflops=tflops)
        
    except subprocess.TimeoutExpired:
        return BenchmarkResult(error="Timeout")
    except Exception as e:
        return BenchmarkResult(error=str(e))


def format_value(value: Optional[float], decimals: int = 4, unit: str = "") -> str:
    """Format a numeric value or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{unit}"


def save_to_csv(results: Dict[str, Dict[str, BenchmarkResult]], 
                benchmarks: list, 
                implementations: list, 
                filename: str):
    """Save benchmark results to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Kernel', 'Implementation', 'Mean Time (ms)', 'Stddev (ms)', 'Performance (TFLOPS)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for kernel, args in benchmarks:
            for impl in implementations:
                result = results[kernel][impl]
                
                if result.is_available():
                    mean_str = f"{result.mean_ms:.4f}" if result.mean_ms is not None else "N/A"
                    stddev_str = f"{result.stddev_ms:.4f}" if result.stddev_ms is not None else "N/A"
                    perf_str = f"{result.tflops:.2f}" if result.tflops is not None else "N/A"
                else:
                    mean_str = "N/A"
                    stddev_str = "N/A"
                    perf_str = "N/A"
                
                writer.writerow({
                    'Kernel': kernel,
                    'Implementation': impl,
                    'Mean Time (ms)': mean_str,
                    'Stddev (ms)': stddev_str,
                    'Performance (TFLOPS)': perf_str
                })
    
    print(f"\nResults saved to: {filename}")


def save_parameters(
    benchmarks: list,
    implementations: list,
    filename: str,
    short_run: bool,
    warmup: int,
    iters: int,
):
    """Save benchmark configuration parameters to a text file."""
    with open(filename, "w") as f:
        f.write("Benchmark Parameters\n")
        f.write("=" * 80 + "\n")
        f.write(f"Implementations: {', '.join(implementations)}\n")
        f.write(f"Short run mode: {'enabled' if short_run else 'disabled'}\n")
        f.write(f"Warmup iterations: {warmup}\n")
        f.write(f"Timed iterations: {iters}\n")
        f.write("\n")

        f.write("Kernel Arguments\n")
        f.write("-" * 80 + "\n")
        for kernel, kernel_args in benchmarks:
            f.write(f"{kernel}: {' '.join(kernel_args)}\n")

    print(f"Parameters saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all kernel benchmarks and collect results into a table"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Output directory for generated files (time.csv, param.txt)"
    )
    parser.add_argument(
        "--short-run",
        action="store_true",
        help="Use warmup=1 and iters=1 for a quick smoke run"
    )
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    default_warmup = 20
    default_iters = 200
    effective_warmup = 1 if args.short_run else default_warmup
    effective_iters = 1 if args.short_run else default_iters
    
    print("=" * 80)
    print("Running All Kernel Benchmarks")
    print("=" * 80)
    if args.short_run:
        print("Mode: short-run (--warmup 1 --iters 1)")
    print()
    
    # Define benchmark configurations
    benchmarks = [
        ("fma", ["--n", "10000000"]),
        ("matmul", ["--m", "1024", "--n", "1024", "--k", "1024"]),
        ("conv2d", ["--n", "1", "--cin", "64", "--cout", "64", "--h", "56", "--w", "56", "--dtype", "fp16"]),
        ("rmsnorm", ["--batch", "4096", "--hidden", "1024"]),
        ("multihead_attention", ["--batch", "16", "--heads", "16", "--seq", "1024", "--head-dim", "64", "--dtype", "fp16"]),
    ]
    
    implementations = ["torch", "triton", "tilelang"]
    # implementations = ["tk"]  
    
    # Store results: results[kernel][impl] = BenchmarkResult
    results: Dict[str, Dict[str, BenchmarkResult]] = {}
    
    # Run all benchmarks
    for kernel, bench_args in benchmarks:
        results[kernel] = {}
        print(f"\n{'=' * 80}")
        print(f"Kernel: {kernel.upper()}")
        print(f"{'=' * 80}")

        run_args = bench_args + ["--warmup", str(effective_warmup), "--iters", str(effective_iters)]
        
        for impl in implementations:
            result = run_benchmark(kernel, impl, run_args)
            results[kernel][impl] = result
            
            if result.is_available():
                status = "✓ SUCCESS"
            else:
                status = f"✗ {result.error}"
            print(f"  {impl:8s}: {status}")
        
        print()
    
    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Create table header
    header = f"{'Kernel':<12} {'Impl':<8} {'Mean (ms)':<12} {'Stddev (ms)':<14} {'Performance':<15}"
    print(header)
    print("-" * len(header))
    
    # Print results for each kernel and implementation
    for kernel, bench_args in benchmarks:
        for impl in implementations:
            result = results[kernel][impl]
            
            if result.is_available():
                mean_str = format_value(result.mean_ms, 4)
                stddev_str = format_value(result.stddev_ms, 4)
                
                if result.tflops is not None:
                    perf_str = format_value(result.tflops, 2, " TFLOPS")
                else:
                    perf_str = "N/A"
            else:
                mean_str = "N/A"
                stddev_str = "N/A"
                perf_str = "N/A"
            
            print(f"{kernel:<12} {impl:<8} {mean_str:<12} {stddev_str:<14} {perf_str:<15}")
    
    print()
    print("=" * 80)
    
    # Generate detailed markdown table
    print("\nMarkdown Table:")
    print()
    print("| Kernel   | Implementation | Mean Time (ms) | Stddev (ms) | Performance      |")
    print("|----------|----------------|----------------|-------------|------------------|")
    
    for kernel, bench_args in benchmarks:
        for impl in implementations:
            result = results[kernel][impl]
            
            if result.is_available():
                mean_str = format_value(result.mean_ms, 4)
                stddev_str = format_value(result.stddev_ms, 4)
                
                if result.tflops is not None:
                    perf_str = format_value(result.tflops, 2) + " TFLOPS"
                else:
                    perf_str = "N/A"
            else:
                mean_str = "N/A"
                stddev_str = "N/A"
                perf_str = "N/A"
            
            print(f"| {kernel:<8} | {impl:<14} | {mean_str:<14} | {stddev_str:<11} | {perf_str:<16} |")
    
    print()
    
    # Save output artifacts
    time_csv_path = os.path.join(args.output_dir, "time.csv")
    param_txt_path = os.path.join(args.output_dir, "param.txt")
    save_to_csv(results, benchmarks, implementations, time_csv_path)
    save_parameters(
        benchmarks,
        implementations,
        param_txt_path,
        args.short_run,
        effective_warmup,
        effective_iters,
    )


if __name__ == "__main__":
    main()
