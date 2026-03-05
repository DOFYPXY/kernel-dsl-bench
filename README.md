# GPU Kernel DSL Benchmarks

Comparing GPU kernel implementations across PyTorch, Triton, and JAX.

**Kernels:** FMA (Fused Multiply-Add), MatMul (Matrix Multiplication), RMSNorm (Root Mean Square Normalization)

**DSLs:** PyTorch, Triton, JAX

## Benchmarked Kernels

### 1. Fused Multiply-Add (FMA)
`y = x * a + b` - Element-wise operation

Directory: `fma/`

### 2. Matrix Multiplication (MatMul)
`C = A @ B` - General matrix multiplication

Directory: `matmul/`

### 3. Root Mean Square Normalization (RMSNorm)
`y = x / sqrt(mean(x^2) + eps)` - Normalization operation

Directory: `rmsnorm/`

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
python run.py fma --impl torch/triton/jax --n 100000000

# MatMul benchmarks
python run.py matmul --impl torch/triton/jax --m 2048 --n 2048 --k 2048

# RMSNorm benchmarks
python run.py rmsnorm --impl torch/triton/jax --batch 4096 --hidden 1024
```

### From Kernel Directory

```bash
cd fma
python benchmark.py --impl torch/triton/jax

# Custom: python benchmark.py --impl triton --n 100000000
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
├── matmul/                # MatMul kernel implementations
│   ├── matmul_torch.py
│   ├── matmul_triton.py
│   ├── matmul_jax.py
│   └── benchmark.py
└── rmsnorm/               # RMSNorm kernel implementations
    ├── rmsnorm_torch.py
    ├── rmsnorm_triton.py
    └── benchmark.py
```

Each kernel directory contains `*_<dsl>.py` files for PyTorch, Triton, and JAX implementations, plus a `benchmark.py` harness.

