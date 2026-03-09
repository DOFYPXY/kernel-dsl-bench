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
- GCC ≥ 9 at `/usr/bin/gcc` (used as the nvcc host compiler — GCC 11.3 was used when developing these kernels)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

---

## Environment Setup

### 1. Clone ThunderKittens

```bash
git clone https://github.com/HazyResearch/ThunderKittens ~/ThunderKittens
```

The kernels in this repo were developed and tested against commit `6fd51f22b5489c544e4f33fb1a21b3d39a79984a` (main branch). To pin to that exact revision:

```bash
git -C ~/ThunderKittens checkout 6fd51f22b5489c544e4f33fb1a21b3d39a79984a
```

Adjust the clone path if you prefer a different location; update `TK_ROOT` accordingly.

### 2. Create Conda Environment

```bash
# (One-time) Install the libmamba SAT solver — vastly faster than conda's classic solver,
# especially for large channels like nvidia and conda-forge.
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

```bash
conda create -n tk_env python=3.10
conda activate tk_env

# Install CUDA 12.4 compiler and runtime only.
# --override-channels + -c nvidia restricts the solve to the nvidia channel,
# preventing conda from mixing in conda-forge package versions for CUDA deps.
# Do NOT add cuda-toolkit — it drags in cuda-cccl 13.x headers that conflict
# with nvcc 12.4 (see note below).
conda install --override-channels -c nvidia \
    "cuda-nvcc=12.4" "cuda-cudart=12.4" "cuda-cudart-dev=12.4" "cuda-cccl=12.4"

# Install Python packages (pinned versions in requirements.txt)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

> **Why libmamba?** conda's classic solver has to explore the full cross-product of all
> package versions across every configured channel. With just `conda-forge` + `nvidia`
> the search space is large enough to stall indefinitely when installing CUDA packages.
> libmamba (C++ SAT solver) solves the same problem in seconds.
>
> **Why `--override-channels -c nvidia`?** Without it, conda also searches `conda-forge`
> for every transitive CUDA dependency. `flexible` channel priority (the default) lets
> conda-forge packages satisfy nvidia deps, multiplying the combinations the solver must
> check. `--override-channels` pins the search to exactly one channel for this one step.
>
> **Why not `cuda-toolkit`?** `cuda-toolkit=12.4` pulls in `cuda-cccl_linux-64 13.x`
> as a transitive dependency. Those CCCL 2.x headers are incompatible with nvcc 12.4 and
> trigger a hard build error: _"CUDA compiler and CUDA toolkit headers are incompatible"_.
> Pinning `cuda-cccl=12.4` explicitly ensures the version-compatible libcudacxx headers
> (including `cuda/pipeline`) are installed without the conflicting 13.x packages.

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
```

> **Note:** `CUDA_INC` is read by each `*_tk.py` JIT loader at import time to pass
> `-I${CUDA_INC}` to `nvcc`. If unset, loaders fall back to
> `$CUDA_HOME/targets/x86_64-linux/include`.

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
-I${CUDA_INC}                          # = $CUDA_HOME/targets/x86_64-linux/include
-I${TK_ROOT}/include
-DNDEBUG
-DKITTENS_AMPERE
--use_fast_math
-DTORCH_COMPILE
```

> **Important — do not add `-I${CUDA_INC}/cccl`**. That subdirectory may contain
> CCCL headers from a newer toolkit version (e.g. 13.x) even when nvcc is 12.4, which
> triggers a hard compile error. nvcc 12.4 resolves its own `cuda/pipeline` and related
> headers internally; no explicit CCCL `-I` path is needed.

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

# Conv2d  (default: N=1, Cin=64, Cout=64, H=56, W=56, 3×3 kernel)
python run.py conv2d --impl tk
python run.py conv2d --impl tk --n 1 --cin 64 --cout 64 --h 56 --w 56

# Multihead Attention  (default: B=16, H=16, S=1024, D=64)
python run.py multihead_attention --impl tk
python run.py multihead_attention --impl tk \
    --batch 16 --heads 16 --seq 1024 --head-dim 64
```

> **Constraints:**
> - MHA `headdim` must be ≤ 64 (fits in one shared memory tile).
> - Triton MHA requires `seqlen` to be a multiple of 32 (`BLOCK_S`).

### From a kernel directory

```bash
cd fma         && python benchmark.py --impl tk --warmup 5 --iters 20
cd matmul      && python benchmark.py --impl tk --m 1024 --n 1024 --k 1024
cd rmsnorm     && python benchmark.py --impl tk
cd conv2d && python benchmark.py --impl tk --n 1 --cin 64 --cout 64 --h 56 --w 56
cd multihead_attention && python benchmark.py --impl tk \
                              --batch 16 --heads 16 --seq 1024 --head-dim 64
```

---

## Benchmark Results (RTX 2080 Ti, sm_75, nvcc 12.4)

Measured with the default configs in `benchmark_all.py` (warmup=20, iters=200):

| Kernel | Config | TK Time | TK Throughput |
|--------|--------|---------|---------------|
| FMA | 10M elements | 0.45 ms | — |
| MatMul | 1024×1024×1024 | 4.74 ms | 0.45 TFLOPS |
| RMSNorm | 4096×1024 | 0.15 ms | — |
| Conv2d | 1×64×56×56, 64×64×3×3 | 0.58 ms | — |
| Multihead Attention | B=16, H=16, S=1024, D=64 | 209.9 ms | 0.33 TFLOPS |

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
