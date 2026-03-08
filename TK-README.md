# ThunderKittens Kernel Setup & Usage

This document covers environment setup, path variables, linker configuration, and benchmark
usage for the [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) (TK) kernel
implementations in this repository.

TK kernels are available for: **FMA**, **MatMul**, **RMSNorm**, **Conv2d**, and
**Multihead Attention** (Flash Attention forward pass).

---

## Prerequisites

- NVIDIA GPU with compute capability ≥ sm_75 (Turing, Ampere, Ada, or Hopper)
- NVIDIA driver ≥ 450. Driver 550.x supports CUDA runtime ≤ 12.4 — see
  [Pinning the CUDA runtime](#pinning-the-cuda-runtime) if this applies to you
- GCC ≥ 9 at `/usr/bin/gcc` (used as the nvcc host compiler)

---

## Environment Setup

### 1. Clone ThunderKittens

```bash
git clone https://github.com/HazyResearch/ThunderKittens ~/ThunderKittens
```

Adjust the path if you prefer a different location; update `TK_ROOT` accordingly.

### 2. Create Conda Environment

```bash
conda create -n tk_env python=3.10
conda activate tk_env

# Install CUDA 12.4 toolkit, headers, and runtime via conda
conda install -c nvidia cuda-nvcc=12.4 cuda-cudart=12.4 cuda-toolkit=12.4

# Install Python packages (pinned versions in requirements.txt)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Pinning the CUDA Runtime

If your NVIDIA driver is 550.x, the maximum supported CUDA runtime is 12.4. Conda may
pull in a newer `libcudart` (e.g. 13.x), which is incompatible and causes TK kernels to
silently output zeros.

Verify and fix the symlink:

```bash
cd $CONDA_PREFIX/lib

# Check current target
ls -la libcudart.so

# Pin to 12.4 if necessary (adjust the .so filename to match what is installed)
ln -sfn libcudart.so.12.4.127 libcudart.so
```

Symptoms of a version mismatch:
- CUDA kernels silently output all zeros
- `CUDA error: no kernel image is available for execution on the device`

### 4. Environment Variables

Add the following to `~/.bashrc` (or your conda env's `activate.d/` script) and
re-source before running TK benchmarks:

```bash
# ThunderKittens include root — adjust if cloned elsewhere
export TK_ROOT=~/ThunderKittens

# CUDA installation (managed by conda)
export CUDA_HOME=$CONDA_PREFIX

# CUDA device include directory used by JIT loaders
# (passed as -I${CUDA_INC} to nvcc during JIT compilation)
export CUDA_INC=$CONDA_PREFIX/targets/x86_64-linux/include

# Target GPU architecture — set to match your hardware:
#   sm_75 = Turing  (RTX 2080 / T4)
#   sm_80 = Ampere  (A100)
#   sm_86 = Ampere  (RTX 3090)
#   sm_89 = Ada     (RTX 4090)
#   sm_90 = Hopper  (H100)
export TORCH_CUDA_ARCH_LIST="7.5"

# Ensure conda-installed tools and libraries are preferred
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Xuanyu added this. It works for some unknown reason.
export NVCC_PREPEND_FLAGS="-DcudaLaunchAttributePreferredClus
terDimension=cudaLaunchAttributeClusterDimension"
```

> **Note:** `CUDA_INC` is read by each `*_tk.py` JIT loader at import time to pass
> `-I${CUDA_INC}` to `nvcc`. If unset, loaders fall back to `$CUDA_HOME/include`.

---

## JIT Compilation

ThunderKittens kernels are compiled on the **first import** via
`torch.utils.cpp_extension.load()`. Compilation takes 1–3 minutes per kernel.

**Cache location:**
```
~/.cache/torch_extensions/py310_cu118/<kernel_name>/
```

**Compiler flags applied by every `*_tk.py` loader:**
```
-std=c++20
-ccbin /usr/bin/gcc
--expt-extended-lambda
--expt-relaxed-constexpr
-gencode arch=compute_75,code=sm_75    # matches TORCH_CUDA_ARCH_LIST
-I${CUDA_INC}
-I${TK_ROOT}/include
-DNDEBUG
-DKITTENS_AMPERE
--use_fast_math
-DTORCH_COMPILE
```

**Forcing a recompile** (e.g. after editing a `.cu` file):
```bash
rm -rf ~/.cache/torch_extensions/py310_cu118/<kernel_name>
```

Replace `<kernel_name>` with one of: `fma_tk`, `matmul_tk`, `rmsnorm_tk`, `conv2d_tk`, `mha_tk`.

---

## Running TK Benchmarks

### From the repository root

```bash
conda activate tk_env

# FMA  (default: 10M elements)
python run.py fma --impl tk
python run.py fma --impl tk --n 10000000 --warmup 5 --iters 20

# MatMul  (default: 1024×1024)
python run.py matmul --impl tk
python run.py matmul --impl tk --m 1024 --n 1024 --k 1024

# RMSNorm  (default: 4096×1024)
python run.py rmsnorm --impl tk

# Conv2d  (default: 8×32×56×56 input, 64×32×3×3 filter)
python run.py conv2d --impl tk
python run.py conv2d --impl tk --batch 8 --c_in 32 --c_out 64 \
    --height 56 --width 56 --kh 3 --kw 3

# Multihead Attention  (default: B=4, H=8, S=512, D=64)
python run.py multihead_attention --impl tk
python run.py multihead_attention --impl tk \
    --batch 4 --heads 8 --seqlen 512 --headdim 64
```

> **Constraints:**
> - MHA `headdim` must be ≤ 64 (fits in one shared memory tile).
> - Triton MHA requires `seqlen` to be a multiple of 32 (`BLOCK_S`).

### From a kernel directory

```bash
cd fma         && python benchmark.py --impl tk --warmup 5 --iters 20
cd matmul      && python benchmark.py --impl tk --m 1024 --n 1024 --k 1024
cd rmsnorm     && python benchmark.py --impl tk
cd conv2d      && python benchmark.py --impl tk \
                      --batch 8 --c_in 32 --c_out 64 --height 56 --width 56
cd multihead_attention && python benchmark.py --impl tk \
                              --batch 4 --heads 8 --seqlen 512 --headdim 64
```

---

## Benchmark Results (RTX 2080 Ti, sm_75)

| Kernel | Config | TK Time | TK Throughput |
|--------|--------|---------|---------------|
| FMA | 10M elements | 0.46 ms | — |
| MatMul | 1024×1024×1024 | 5.96 ms | 0.36 TFLOPS |
| RMSNorm | 4096×1024 | 0.16 ms | — |
| Conv2d | 8×32×56×56, 64×32×3×3 | 0.74 ms | 1.24 TFLOPS |
| Multihead Attention | B=4, H=8, S=512, D=64 | 8.13 ms | 0.26 TFLOPS |

---

## Project Structure (TK files)

```
kernel-dsl-bench/
├── requirements.txt                 # Pinned Python dependencies
├── TK-README.md                     # This file
├── fma/
│   ├── fma_tk.cu                    # ThunderKittens CUDA kernel
│   └── fma_tk.py                    # JIT loader
├── matmul/
│   ├── matmul_tk.cu
│   └── matmul_tk.py
├── rmsnorm/
│   ├── rmsnorm_tk.cu
│   └── rmsnorm_tk.py
├── conv2d/
│   ├── conv2d_tk.cu
│   └── conv2d_tk.py
└── multihead_attention/
    ├── multihead_attention_tk.cu
    └── multihead_attention_tk.py
```
