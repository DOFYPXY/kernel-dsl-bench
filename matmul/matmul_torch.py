"""PyTorch implementation of Matrix Multiplication (MatMul) kernel: C = A @ B"""

import torch


def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch baseline matrix multiplication: C = A @ B
    
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
        
    Returns:
        Output matrix C of shape (M, N) where C = A @ B
    """
    return torch.matmul(a, b)
