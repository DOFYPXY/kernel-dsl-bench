"""
PyTorch baseline implementation of Conv2D (NCHW).
"""

import torch
import torch.nn.functional as F


def torch_conv2d(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 1,
) -> torch.Tensor:
    """
    Conv2D baseline using torch.nn.functional.conv2d.

    Args:
        x: Input tensor, shape (N, C_in, H, W), CUDA
        w: Weight tensor, shape (C_out, C_in, K_h, K_w), CUDA
        bias: Optional bias, shape (C_out,)
        stride: int (currently benchmark will use 1)
        padding: int (currently benchmark will use 1)

    Returns:
        y: Output tensor, shape (N, C_out, H_out, W_out)
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    return F.conv2d(x, w, bias=bias, stride=stride, padding=padding)