"""Matrix Multiplication (MatMul) kernel implementations across different GPU DSLs."""

from .matmul_torch import torch_matmul
from .matmul_triton import triton_matmul

try:
    from .matmul_jax import jax_matmul
except (ImportError, AttributeError):
    jax_matmul = None

__all__ = ["torch_matmul", "triton_matmul", "jax_matmul"]
