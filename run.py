#!/usr/bin/env python3
"""
Benchmark runner script - run benchmarks from root directory.

Usage:
    python run.py fma --impl torch
    python run.py matmul --impl triton --m 2048 --n 2048 --k 2048
"""

import os
import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <kernel> [benchmark_args...]")
        print("")
        print("Kernels:")
        print("  fma     - Fused Multiply-Add")
        print("  matmul  - Matrix Multiplication")
        print("")
        print("Examples:")
        print("  python run.py fma --impl torch")
        print("  python run.py matmul --impl triton --m 2048 --n 2048 --k 2048")
        sys.exit(1)
    
    kernel = sys.argv[1]
    args = sys.argv[2:]
    
    # Validate kernel name
    if kernel not in ["fma", "matmul"]:
        print(f"Error: Unknown kernel '{kernel}'")
        print("Valid kernels: fma, matmul")
        sys.exit(1)
    
    # Get root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_dir = os.path.join(root_dir, kernel)
    
    # Check if kernel directory exists
    if not os.path.isdir(kernel_dir):
        print(f"Error: Kernel directory '{kernel_dir}' not found")
        sys.exit(1)
    
    # Change to kernel directory and run benchmark
    try:
        result = subprocess.run(
            [sys.executable, "benchmark.py"] + args,
            cwd=kernel_dir
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
