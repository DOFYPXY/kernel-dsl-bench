"""JAX implementation of Matrix Multiplication (MatMul) kernel: C = A @ B"""

import jax
import jax.numpy as jnp
import torch


@jax.jit
def jax_matmul_jitted(a, b):
    """
    JIT-compiled JAX matrix multiplication.
    
    Args:
        a: Input matrix A
        b: Input matrix B
        
    Returns:
        Output matrix C = A @ B
    """
    return jnp.matmul(a, b)


def jax_matmul(a_torch: torch.Tensor, b_torch: torch.Tensor) -> torch.Tensor:
    """
    JAX implementation wrapper for matrix multiplication.
    
    Converts torch tensors to JAX, performs computation with JIT,
    and converts result back to torch.
    
    Args:
        a_torch: Input matrix A of shape (M, K) (on CUDA/GPU)
        b_torch: Input matrix B of shape (K, N) (on CUDA/GPU)
        
    Returns:
        Output matrix C of shape (M, N) where C = A @ B (as torch tensor)
    """
    # Convert to JAX (zero-copy on GPU via DLPack)
    a_jax = jnp.asarray(a_torch)
    b_jax = jnp.asarray(b_torch)
    
    # Compute with JIT compilation
    c_jax = jax_matmul_jitted(a_jax, b_jax)
    
    # Convert back to torch
    return torch.from_dlpack(c_jax)
