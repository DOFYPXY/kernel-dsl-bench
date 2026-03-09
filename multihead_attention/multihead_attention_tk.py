"""
ThunderKittens implementation of Scaled Dot-Product Multihead Attention:
  Out = softmax(Q * K^T / sqrt(D)) * V

The CUDA kernel (multihead_attention_tk.cu) is JIT-compiled on first import via
torch.utils.cpp_extension.load.

Algorithm
---------
Flash Attention forward pass on sm_75 (RTX 2080 Ti / Turing).
- Each warp handles BLOCK_S=32 query rows for one (batch, head).
- Streams over K/V in BLOCK_S=32 tiles with online softmax.
- Inner QK^T and PV products use fp32 scalar arithmetic (no tensor cores on sm_75).
- Constraint: head_dim D ≤ 64 (MHA_BLOCK_D).

Environment variables (with sensible defaults):
  TK_ROOT   – path to ThunderKittens repo  (default: ~/ThunderKittens)
  CUDA_HOME – path to CUDA toolkit          (default: conda tk_env)
"""

import math
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

    src = os.path.join(_DIR, "multihead_attention_tk.cu")
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
        name="mha_tk",
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


def tk_multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    ThunderKittens multihead attention: Out = softmax(QK^T / sqrt(D)) * V.

    Args:
        q: (B, H, S, D) float32 CUDA tensor, D ≤ 64
        k: (B, H, S, D) float32 CUDA tensor
        v: (B, H, S, D) float32 CUDA tensor
        causal: whether to apply causal masking (currently unsupported)

    Returns:
        (B, H, S, D) float32 CUDA tensor
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if causal:
        raise NotImplementedError("tk_multihead_attention currently supports causal=False only")

    D = q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    return _get_module().mha_fwd(q, k, v, scale)
