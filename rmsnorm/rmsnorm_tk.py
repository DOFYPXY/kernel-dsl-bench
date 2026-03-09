"""
ThunderKittens implementation of RMSNorm:
  y = (x / sqrt(mean(x^2) + eps)) * weight

The CUDA kernel (rmsnorm_tk.cu) is JIT-compiled on first import via
torch.utils.cpp_extension.load.  Compilation requires:
  • nvcc ≥ 12.x with C++20 support  (conda install -c nvidia cuda-nvcc)
  • ThunderKittens headers at TK_ROOT  (git clone HazyResearch/ThunderKittens)

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

    src = os.path.join(_DIR, "rmsnorm_tk.cu")
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
        name="rmsnorm_tk",
        sources=[src],
        extra_cuda_cflags=[
            "-std=c++20",
            "-ccbin", "/usr/bin/gcc",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-DcudaLaunchAttributePreferredClusterDimension=cudaLaunchAttributeClusterDimension",
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


def tk_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    ThunderKittens RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight

    Args:
        x:      2-D float32 CUDA tensor of shape (batch, hidden)
        weight: 1-D float32 CUDA tensor of shape (hidden,)
        eps:    numerical stability constant

    Returns:
        Tensor of shape (batch, hidden) with same dtype/device as x
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA device"
    assert x.is_contiguous() and weight.is_contiguous(), \
        "Tensors must be contiguous"
    assert x.shape[-1] == weight.numel(), \
        "Last dim of x must match weight length"

    return _get_module().rmsnorm(x, weight, float(eps))
