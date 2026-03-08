"""
ThunderKittens implementation of 2D Convolution: y = conv2d(x, weight)

The CUDA kernel (conv2d_tk.cu) is JIT-compiled on first import via
torch.utils.cpp_extension.load.  Compilation requires:
  • nvcc ≥ 12.x with C++20 support
  • ThunderKittens headers at TK_ROOT  (git clone HazyResearch/ThunderKittens)

Algorithm
---------
Direct tiled convolution on sm_75 (RTX 2080 Ti / Turing).

Each warp handles TILE_OHW=64 output positions × TILE_OC=16 output channels.
For every (C_in, KH, KW) combination:
  1. Load TK_TILE_OC weight values into shared memory
  2. Load TK_TILE_OHW input values (gathered from padded input) into shared memory
  3. Each thread accumulates the outer product of its 2 input values × 16 weight values

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

    src = os.path.join(_DIR, "conv2d_tk.cu")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Cannot find kernel source: {src}")
    if not os.path.isdir(_TK_ROOT):
        raise FileNotFoundError(
            f"ThunderKittens not found at {_TK_ROOT}. "
            "Set TK_ROOT or: git clone https://github.com/HazyResearch/ThunderKittens"
        )

    os.environ.setdefault("CUDA_HOME", _CUDA_HOME)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

    _CUDA_INC = os.path.join(_CUDA_HOME, "targets", "x86_64-linux", "include")

    # Conda CUDA packages usually place libcudart in $CUDA_HOME/lib (not lib64).
    _cuda_ldflags = []
    for libdir in (os.path.join(_CUDA_HOME, "lib"), os.path.join(_CUDA_HOME, "lib64")):
        if os.path.isdir(libdir):
            _cuda_ldflags.extend([f"-L{libdir}", f"-Wl,-rpath,{libdir}"])

    import site
    nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
    pip_nvidia_incs = []
    if os.path.isdir(nvidia_base):
        for pkg in sorted(os.listdir(nvidia_base)):
            inc = os.path.join(nvidia_base, pkg, "include")
            if os.path.isdir(inc):
                pip_nvidia_incs.append(f"-I{inc}")

    _module = load(
        name="conv2d_tk",
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


def tk_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    ThunderKittens 2D convolution: y = conv2d(x, weight, bias=None, stride=1, padding=0).

    Args:
        x:       (N, C_in, H, W) float32 CUDA tensor
        weight:  (C_out, C_in, KH, KW) float32 CUDA tensor
        bias:    (C_out,) float32 CUDA tensor, optional
        stride:  convolution stride (default: 1)
        padding: zero-padding (default: 0)

    Returns:
        (N, C_out, OH, OW) float32 CUDA tensor
    """
    x      = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    return _get_module().conv2d(x, weight, bias, stride, padding)
