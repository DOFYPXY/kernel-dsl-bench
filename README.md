# GPU Kernel DSL Benchmarks

Comparing GPU kernel implementations across PyTorch, Triton, and JAX.

**Kernels:** FMA (Fused Multiply-Add), MatMul (Matrix Multiplication)

**DSLs:** PyTorch, Triton, JAX

## Benchmarked Kernels

### 1. Fused Multiply-Add (FMA)
`y = x * a + b` - Element-wise operation

Directory: `fma/`

### 2. Matrix Multiplication (MatMul)
`C = A @ B` - General matrix multiplication

Directory: `matmul/`

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .              # PyTorch + Triton
pip install -e ".[jax]"       # Add JAX support
```

## Running Benchmarks

### From Root Directory (Recommended)
```bash
# FMA benchmarks
python run.py fma --impl torch
python run.py fma --impl triton --n 100000000 --dtype fp32

# MatMul benchmarks
python run.py matmul --impl torch
python run.py matmul --impl triton --m 2048 --n 2048 --k 2048
```

### From Kernel Directory

#### FMA Benchmark
```bash
cd fma
python benchmark.py --impl torch     # PyTorch
python benchmark.py --impl triton    # Triton
python benchmark.py --impl jax       # JAX

# Custom: python benchmark.py --impl triton --n 100000000 --dtype fp32
```

#### MatMul Benchmark
```bash
cd matmul
python benchmark.py --impl torch     # PyTorch
python benchmark.py --impl triton    # Triton
python benchmark.py --impl jax       # JAX

# Custom: python benchmark.py --impl triton --m 2048 --n 2048 --k 2048
```

## Project Structure

```
kernel-dsl/
├── common.py              # Shared utilities
├── fma/                   # FMA kernel implementations
│   ├── fma_torch.py
│   ├── fma_triton.py
│   ├── fma_jax.py
│   └── benchmark.py
└── matmul/                # MatMul kernel implementations
    ├── matmul_torch.py
    ├── matmul_triton.py
    ├── matmul_jax.py
    └── benchmark.py
```

Each kernel directory contains `*_<dsl>.py` files for PyTorch, Triton, and JAX implementations, plus a `benchmark.py` harness.

