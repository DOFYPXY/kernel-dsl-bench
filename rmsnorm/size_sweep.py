#!/usr/bin/env python3
"""
Sweep RMSNorm hidden dimensions across multiple kernel DSL implementations.

Default sweep uses 16 linearly increasing hidden dimensions from:
- hidden = 128
- hidden = 2048

Batch size is calculated as: batch = hidden * 4

Run from repository root:
    python rmsnorm/size_sweep.py

Or from rmsnorm/:
    python size_sweep.py
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

# Determine repository root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT_DIR)
from common import build_subprocess_env  # noqa: E402


def linear_sizes(min_size: int, step_size: int, num_sizes: int) -> List[int]:
    """Generate linearly increasing integer sizes with a fixed step."""
    return [min_size + i * step_size for i in range(num_sizes)]


def run_rmsnorm_benchmark(hidden: int, batch: int, impl: str, warmup: int, iters: int) -> Dict[str, Optional[float]]:
    """Run rmsnorm benchmark via subprocess and parse output."""
    try:
        result = subprocess.run(
            [sys.executable, "run.py", "rmsnorm", "--impl", impl, "--hidden", str(hidden),
             "--batch", str(batch), "--warmup", str(warmup), "--iters", str(iters)],
            capture_output=True, text=True, timeout=120, cwd=_ROOT_DIR,
            env=build_subprocess_env())
        
        output = result.stdout + result.stderr
        if result.returncode != 0:
            if "not implemented" in output.lower() or "not available" in output.lower():
                return {"error": "Not implemented"}
            return {"error": f"Exit code {result.returncode}"}
        
        # Parse output
        mean = re.search(r'Mean time:\s+([\d.]+)\s+ms', output)
        stddev = re.search(r'Stddev:\s+([\d.]+)\s+ms', output)
        tflops = re.search(r'Performance:\s+([\d.]+)\s+TFLOPS', output)
        
        if not mean:
            return {"error": "Failed to parse output"}
        
        return {"mean_ms": float(mean.group(1)),
                "stddev_ms": float(stddev.group(1)) if stddev else None,
                "tflops": float(tflops.group(1)) if tflops else None,
                "error": None}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}


def save_parameters(filename: str, impls: List[str], sizes: List[int], warmup: int, iters: int, seed: int) -> None:
    """Save size sweep parameters to a text file."""
    with open(filename, "w") as f:
        f.write(f"RMSNorm Size Sweep Parameters\n{'=' * 80}\n"
                f"Implementations: {', '.join(impls)}\n"
                f"Warmup iterations: {warmup}\nTimed iterations: {iters}\nSeed: {seed}\n\n"
                f"Hidden dimensions ({len(sizes)}): {sizes}\n"
                f"Batch size formula: batch = hidden * 4\n")


def save_times_csv(filename: str, rows: List[Dict[str, str]]) -> None:
    """Save benchmark rows to CSV."""
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["impl", "hidden", "batch", "mean_ms", "stddev_ms", "tflops", "status"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep RMSNorm hidden dimensions over multiple DSL implementations"
    )
    parser.add_argument(
        "--min-hidden",
        type=int,
        default=128,
        help="Minimum hidden dimension (default: 128)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=128,
        help="Step size between consecutive hidden values (default: 128)",
    )
    parser.add_argument(
        "--num-sizes",
        type=int,
        default=16,
        help="Number of hidden dimensions to sample (default: 16)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Timed iterations (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Output directory for generated files (time.csv, param.txt)",
    )
    args = parser.parse_args()

    if args.min_hidden <= 0 or args.step_size <= 0 or args.num_sizes <= 0:
        raise ValueError("Size parameters must be positive")

    # implementations = ["torch", "triton", "tilelang"]
    implementations = ["tk"]
    
    hidden_sizes = linear_sizes(args.min_hidden, args.step_size, args.num_sizes)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'=' * 80}\nRMSNorm Size Sweep\n{'=' * 80}")
    print(f"Implementations: {', '.join(implementations)}")
    print(f"Hidden dimensions ({len(hidden_sizes)}): {hidden_sizes}\n"
          f"Batch size formula: batch = hidden * 4\nWarmup: {args.warmup}, Timed iters: {args.iters}\n")

    rows: List[Dict[str, str]] = []

    for hidden in hidden_sizes:
        batch = hidden * 4
        print(f"Hidden={hidden}, Batch={batch}")

        for impl in implementations:
            result = run_rmsnorm_benchmark(hidden, batch, impl, args.warmup, args.iters)
            
            if result.get("error"):
                print(f"  {impl:8s}: {result['error']}")
                row = {"impl": impl, "hidden": str(hidden), "batch": str(batch),
                       "mean_ms": "N/A", "stddev_ms": "N/A",
                       "tflops": "N/A", "status": result["error"]}
            else:
                mean, std, tf = result["mean_ms"], result["stddev_ms"], result["tflops"]
                tflops_str = f"{tf:.2f} TFLOPS" if tf else "N/A"
                print(f"  {impl:8s}: {mean:8.4f} ms  {std:8.4f} ms  {tflops_str}")
                row = {"impl": impl, "hidden": str(hidden), "batch": str(batch),
                       "mean_ms": f"{mean:.4f}", "stddev_ms": f"{std:.4f}" if std else "N/A",
                       "tflops": f"{tf:.2f}" if tf else "N/A", "status": "OK"}
            rows.append(row)

        print()

    print(f"\nSummary\n{'=' * 80}")
    header = f"{'Impl':<10} {'Hidden':>8} {'Batch':>8} {'Mean(ms)':>10} {'Std(ms)':>10} {'TFLOPS':>8} {'Status':<10}"
    print(f"{header}\n{'-' * len(header)}")
    for row in rows:
        print(f"{row['impl']:<10} {row['hidden']:>8} {row['batch']:>8} "
              f"{row['mean_ms']:>10} {row['stddev_ms']:>10} "
              f"{row['tflops']:>8} {row['status']:<10}")

    save_times_csv(os.path.join(args.output_dir, "time_tk.csv"), rows)
    save_parameters(os.path.join(args.output_dir, "param_tk.txt"), implementations, hidden_sizes,
                    args.warmup, args.iters, args.seed)
    print(f"\nSaved to: {args.output_dir}/time_tk.csv and {args.output_dir}/param_tk.txt")


if __name__ == "__main__":
    main()
