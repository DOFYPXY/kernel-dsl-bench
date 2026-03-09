"""Fused Multiply-Add (FMA) kernel implementations across different GPU DSLs."""

from .fma_torch import torch_fma
from .fma_triton import triton_fma

# Try to import jax if available
try:
    from .fma_jax import jax_fma
except (ImportError, AttributeError):
    jax_fma = None

# Try to import tilelang if available
try:
    from .fma_tilelang import tilelang_fma
except (ImportError, AttributeError):
    tilelang_fma = None

__all__ = ["torch_fma", "triton_fma", "jax_fma", "tilelang_fma"]
