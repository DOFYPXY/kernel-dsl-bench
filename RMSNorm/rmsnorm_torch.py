"""
PyTorch implementation of RMSNorm.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
Normalization is applied over the last dimension.
"""

import torch


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    PyTorch baseline RMSNorm.

    Args:
        x: Input tensor of shape (..., hidden), on CUDA
        weight: Scale tensor of shape (hidden,), on same device/dtype as x
        eps: Small constant for numerical stability

    Returns:
        Tensor with same shape/dtype/device as x
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert weight.is_cuda, "Weight must be on CUDA device"
    assert x.shape[-1] == weight.numel(), "Last dim of x must match weight length"

    # Compute rms over last dimension
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    y = x / rms

    # Apply per-channel scale (broadcast over leading dims)
    return y * weight