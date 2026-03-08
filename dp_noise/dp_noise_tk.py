"""
ThunderKittens implementation of epsilon-Differential Privacy (Laplace mechanism).

    y = clip(x, clip_norm) + Laplace(0, clip_norm / epsilon)

The CUDA kernel (dp_noise_tk.cu) is JIT-compiled on first import via
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

    src = os.path.join(_DIR, "dp_noise_tk.cu")
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

    # pip nvidia package headers (cusparse.h, cublas.h, etc.)
    import site
    nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
    pip_nvidia_incs = []
    if os.path.isdir(nvidia_base):
        for pkg in sorted(os.listdir(nvidia_base)):
            inc = os.path.join(nvidia_base, pkg, "include")
            if os.path.isdir(inc):
                pip_nvidia_incs.append(f"-I{inc}")

    _module = load(
        name="dp_noise_tk",
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


def tk_dp_noise(
    x: torch.Tensor,
    epsilon: float,
    clip_norm: float,
    seed: int = 42,
) -> torch.Tensor:
    """
    ThunderKittens DP Laplace mechanism:
        y = clip(x, clip_norm) + Laplace(0, clip_norm / epsilon)

    The full pipeline runs on-device (two TK kernels, no host sync):
      1. Compute ‖x‖₂² via tile-level accumulation + warp reduction + atomicAdd
      2. Clip + generate Laplace noise via cuRAND Philox in register tiles

    Args:
        x:         1-D float32 CUDA tensor
        epsilon:   privacy budget  (ε > 0)
        clip_norm: L2 clipping bound  (C > 0)
        seed:      RNG seed for reproducibility

    Returns:
        1-D float32 tensor of the same shape as x
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.dtype == torch.float32, "Input must be float32"
    assert epsilon > 0, "epsilon must be positive"
    assert clip_norm > 0, "clip_norm must be positive"

    return _get_module().dp_noise(x, float(epsilon), float(clip_norm), int(seed))
