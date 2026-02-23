"""Triton implementation of Fused Multiply-Add (FMA) kernel: y = x * a + b"""

import torch
import triton
import triton.language as tl


@triton.jit
def triton_fma_kernel(
    y_ptr,
    x_ptr,
    a,
    b,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel: y = x * a + b (element-wise fused multiply-add).
    
    Args:
        y_ptr: Pointer to output tensor
        x_ptr: Pointer to input tensor
        a: Scalar multiplier
        b: Scalar addend
        n_elements: Number of elements to process
        BLOCK_SIZE: Number of elements per block (compile-time constant)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Load, compute, and store
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * a + b
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_fma(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    Triton implementation wrapper for FMA.
    
    Args:
        x: Input tensor (must be on CUDA and contiguous)
        a: Scalar multiplier
        b: Scalar addend
        
    Returns:
        Output tensor y = x * a + b
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.is_contiguous(), "Input must be contiguous"
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    y = torch.empty_like(x)
    triton_fma_kernel[grid](y, x, a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return y
