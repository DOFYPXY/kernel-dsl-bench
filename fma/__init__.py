"""Fused Multiply-Add (FMA) kernel implementations across different GPU DSLs."""

from .fma_torch import torch_fma
from .fma_triton import triton_fma
from .fma_jax import jax_fma

__all__ = ["torch_fma", "triton_fma", "jax_fma"]
