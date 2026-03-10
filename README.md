# GPU Kernel DSL Benchmarks

Comparing GPU kernel implementations across PyTorch, Triton, JAX, and ThunderKittens.

**Kernels:** FMA (Fused Multiply-Add), MatMul (Matrix Multiplication), RMSNorm (Root Mean Square Normalization), Multihead Attention

**DSLs:** PyTorch, Triton, JAX, ThunderKittens (TK)

## Benchmarked Kernels

### 1. Fused Multiply-Add (FMA)
`y = x * a + b` - Element-wise operation

### 2. Matrix Multiplication (MatMul)
`C = A @ B` - General matrix multiplication

### 3. Root Mean Square Normalization (RMSNorm)
`y = x / sqrt(mean(x^2) + eps)` - Normalization operation

### 4. Multihead Attention
Scaled dot-product attention over multiple heads.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
# Install dependencies
pip install -e .              # Only for PyTorch + Triton + Tilelang
```

## Generate Final Data

### Notes
Manually switch the implementation in `benchmark_all.py`, `matmul/size_sweep.py`, and `rmsnorm/size_sweep.py`:
```python
    # implementations = ["torch", "triton", "tilelang"]
    implementations = ["tk"]  
```
The Tilelang kernels of Conv2D and MHA do not support FP32. Manually set `--dtype 16` in default parameters in `benchmark_all.py`.


### Comparison over fixed parameters
```bash
python benchmark_all.py
python benchmark_all.py -o results/
# Visualization
python scripts/visualize.py results/time.csv 
```

### Size sweep (for Matmul + RMSNorm)

Matmul
```bash
# Run
python matmul/size_sweep.py --num-sizes 16 --min-size 128 --step-size 128 -o results/matmul_size/ 
# Visualization
python scripts/matmul_size_curves.py results/matmul_size/time.csv
```
RMSNorm
```bash
# Run
python rmsnorm/size_sweep.py --min-hidden 128 --step-size 128 --num-sizes 16 -o results/rmsnorm_size
# Visualization
python scripts/rmsnorm_size_curves.py results/rmsnorm_size/time.csv
```


## Running Benchmarks

### From Root Directory (Recommended)
```bash
# FMA benchmarks
python run.py fma --impl torch/triton/jax/tk --n 100000000

# MatMul benchmarks
python run.py matmul --impl torch/triton/jax/tk --m 2048 --n 2048 --k 2048

# RMSNorm benchmarks
python run.py rmsnorm --impl torch/triton/jax/tk --batch 4096 --hidden 1024

# Multihead Attention benchmarks
python run.py multihead_attention --impl torch/triton/tk --batch 16 --heads 16 --seq 1024 --head-dim 64

# Conv2D benchmarks
python run.py conv2d --impl torch/triton/tk --n 1 --cin 64 --cout 64 --h 56 --w 56
```

### From Kernel Directory

```bash
cd fma
python benchmark.py --impl torch/triton/jax

# Custom: python benchmark.py --impl tk --n 100000000
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
├── multihead_attention/   # Multihead attention kernel implementations
│   ├── multihead_attention_torch.py
│   └── benchmark.py
```

Each kernel directory contains `*_<dsl>.py` files for PyTorch, Triton, and JAX implementations, plus a `benchmark.py` harness.
