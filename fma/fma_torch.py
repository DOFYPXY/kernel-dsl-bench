"""PyTorch implementation of Fused Multiply-Add (FMA) kernel: y = x * a + b"""

import torch


def torch_fma(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    PyTorch baseline FMA: y = x * a + b
    
    Args:
        x: Input tensor
        a: Scalar multiplier
        b: Scalar addend
        
    Returns:
        Output tensor y = x * a + b
    """
    return x * a + b
