"""PyTorch baseline implementation of 2D convolution."""

import torch
import torch.nn.functional as F


def torch_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    2D convolution using PyTorch.

    Args:
        x:      (N, C_in, H, W) float32 CUDA tensor
        weight: (C_out, C_in, KH, KW) float32 CUDA tensor
        bias:   (C_out,) float32 CUDA tensor, optional
        stride: stride for height and width
        padding: zero-padding for height and width

    Returns:
        (N, C_out, OH, OW) float32 CUDA tensor
    """
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding)
