"""
ThunderKittens implementation of Matrix Multiplication: C = A @ B

The CUDA kernel (matmul_tk.cu) is JIT-compiled on first import via
torch.utils.cpp_extension.load.  Compilation requires:
  • nvcc ≥ 12.x with C++20 support  (conda install -c nvidia cuda-nvcc)
  • ThunderKittens headers at TK_ROOT  (git clone HazyResearch/ThunderKittens)

Architecture note
-----------------
TK's warp::mma_AB needs sm_80+ (Ampere). On sm_75 (Turing / RTX 2080 Ti) this
kernel instead uses TK tile descriptors for efficient HBM↔SMEM movement and
a hand-written fp32 inner product, achieving shared-memory tiled GEMM performance.

Environment variables (with sensible defaults):
  TK_ROOT   – path to ThunderKittens repo  (default: ~/ThunderKittens)
  CUDA_HOME – path to CUDA toolkit          (default: conda tk_env)
"""

import os
import torch
from torch.utils.cpp_extension import load

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_TK_ROOT   = os.environ.get("TK_ROOT",
             os.path.expanduser("~/ThunderKittens"))
_CUDA_HOME = os.environ.get("CUDA_HOME",
             "/home/ubuntu/.conda/envs/tk_env")

_module = None

def _get_module():
    global _module
    if _module is not None:
        return _module

    src = os.path.join(_DIR, "matmul_tk.cu")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Cannot find kernel source: {src}")
    if not os.path.isdir(_TK_ROOT):
        raise FileNotFoundError(
            f"ThunderKittens not found at {_TK_ROOT}. "
            "Set TK_ROOT or: git clone https://github.com/HazyResearch/ThunderKittens"
        )

    os.environ.setdefault("CUDA_HOME", _CUDA_HOME)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

    # The conda tk_env has CUDA 13.1 nvcc + headers, but PyTorch ships
    # pip nvidia-cuda-runtime-cu11 with CUDA 11.8 headers.  Placing the
    # conda CUDA include path FIRST (-I beats -isystem) resolves the conflict.
    _CUDA_INC = os.path.join(_CUDA_HOME, "targets", "x86_64-linux", "include")

    # Conda CUDA packages usually place libcudart in $CUDA_HOME/lib (not lib64).
    _cuda_ldflags = []
    for libdir in (os.path.join(_CUDA_HOME, "lib"), os.path.join(_CUDA_HOME, "lib64")):
        if os.path.isdir(libdir):
            _cuda_ldflags.extend([f"-L{libdir}", f"-Wl,-rpath,{libdir}"])

    # Also need pip nvidia package headers (cusparse.h, cublas.h, etc.)
    import site
    nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
    pip_nvidia_incs = []
    if os.path.isdir(nvidia_base):
        for pkg in sorted(os.listdir(nvidia_base)):
            inc = os.path.join(nvidia_base, pkg, "include")
            if os.path.isdir(inc):
                pip_nvidia_incs.append(f"-I{inc}")

    _module = load(
        name="matmul_tk",
        sources=[src],
        extra_cuda_cflags=[
            "-std=c++20",
            "-ccbin", "/usr/bin/gcc",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-gencode", "arch=compute_75,code=sm_75",
            f"-I{_CUDA_INC}",
            f"-I{_CUDA_INC}/cccl",
            f"-I{_TK_ROOT}/include",
            "-DNDEBUG",
            "-DKITTENS_AMPERE",
            "--use_fast_math",
            "-DTORCH_COMPILE",
        ] + pip_nvidia_incs,
        extra_ldflags=_cuda_ldflags,
        verbose=False,
    )
    return _module


def tk_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    ThunderKittens MatMul: C = A @ B  (fp32, CUDA).

    Args:
        a: 2-D float32 CUDA tensor of shape (M, K)
        b: 2-D float32 CUDA tensor of shape (K, N)

    Returns:
        2-D float32 tensor of shape (M, N)
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA device"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"

    return _get_module().matmul(a, b)
