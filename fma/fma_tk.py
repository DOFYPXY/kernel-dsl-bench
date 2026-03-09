"""
ThunderKittens implementation of Fused Multiply-Add (FMA): y = x * a + b

The CUDA kernel (fma_tk.cu) is JIT-compiled on first import via
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

    src = os.path.join(_DIR, "fma_tk.cu")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Cannot find kernel source: {src}")
    if not os.path.isdir(_TK_ROOT):
        raise FileNotFoundError(
            f"ThunderKittens not found at {_TK_ROOT}. "
            "Set TK_ROOT or: git clone https://github.com/HazyResearch/ThunderKittens"
        )

    # Set CUDA_HOME so torch.utils.cpp_extension picks up the conda nvcc.
    # Pin arch to sm_75 (RTX 2080 Ti / Turing) to avoid auto-detection issues
    # when the conda nvcc reports a newer CUDA toolkit version.
    os.environ.setdefault("CUDA_HOME", _CUDA_HOME)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

    # The conda tk_env has CUDA 13.1 nvcc + headers, but PyTorch ships
    # pip nvidia-cuda-runtime-cu11 with CUDA 11.8 headers.  Placing the conda
    # CUDA include path FIRST (-I beats -isystem) resolves the conflict.
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

    # cudaLaunchAttributePreferredClusterDimension was introduced in CUDA 12.0.
    # On minimal conda installs the CUDA 12.4 driver_types.h may not be on the
    # search path yet, so TK's util.cuh fails to find it. Detect and add a
    # fallback macro only when the symbol is absent from the installed headers.
    _driver_types = os.path.join(_CUDA_INC, "driver_types.h")
    _has_preferred_cluster = (
        os.path.isfile(_driver_types)
        and "cudaLaunchAttributePreferredClusterDimension" in open(_driver_types).read()
    )
    _preferred_cluster_flag = [] if _has_preferred_cluster else [
        "-DcudaLaunchAttributePreferredClusterDimension=cudaLaunchAttributeClusterDimension"
    ]

    _module = load(
        name="fma_tk",
        sources=[src],
        extra_cuda_cflags=[
            "-std=c++20",
            "-ccbin", "/usr/bin/gcc",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-gencode", "arch=compute_75,code=sm_75",
            f"-I{_CUDA_INC}",
            f"-I{_TK_ROOT}/include",
            "-DNDEBUG",
            "-DKITTENS_AMPERE",
            "--use_fast_math",
            "-DTORCH_COMPILE",
        ] + _preferred_cluster_flag + pip_nvidia_incs,
        extra_ldflags=_cuda_ldflags,
        verbose=False,
    )
    return _module


def tk_fma(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    ThunderKittens FMA: y = x * a + b  (element-wise, fp32, CUDA).

    Args:
        x: 1-D float32 CUDA tensor
        a: scalar multiplier
        b: scalar addend

    Returns:
        1-D float32 tensor of the same shape as x
    """
    assert x.is_cuda,       "Input must be on CUDA device"
    assert x.is_contiguous(), "Input must be contiguous"

    return _get_module().fma(x, float(a), float(b))
