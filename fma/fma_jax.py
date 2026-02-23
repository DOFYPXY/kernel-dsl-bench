"""JAX implementation of Fused Multiply-Add (FMA) kernel: y = x * a + b"""

import jax
import jax.numpy as jnp
import torch


@jax.jit
def jax_fma_jitted(x, a, b):
    """
    JIT-compiled JAX FMA kernel.
    
    Args:
        x: Input array
        a: Scalar multiplier
        b: Scalar addend
        
    Returns:
        Output array y = x * a + b
    """
    return x * a + b


def jax_fma(x_torch: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    JAX implementation wrapper for FMA.
    
    Converts torch tensor to JAX, performs computation with JIT,
    and converts result back to torch.
    
    Args:
        x_torch: Input tensor (on CUDA/GPU)
        a: Scalar multiplier
        b: Scalar addend
        
    Returns:
        Output tensor y = x * a + b (as torch tensor)
    """
    # Convert to JAX (zero-copy on GPU via DLPack)
    x_jax = jnp.asarray(x_torch)
    
    # Compute with JIT compilation
    y_jax = jax_fma_jitted(x_jax, a, b)
    
    # Convert back to torch
    return torch.from_dlpack(y_jax)
