"""PyTorch baseline implementation of Multihead Attention."""

import torch
import torch.nn.functional as F
import math


def torch_multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product multihead attention: Out = softmax(QK^T / sqrt(d)) * V

    Args:
        q: (B, H, S, D) float32 CUDA tensor  (queries)
        k: (B, H, S, D) float32 CUDA tensor  (keys)
        v: (B, H, S, D) float32 CUDA tensor  (values)
        scale: attention scale (default: 1/sqrt(D))

    Returns:
        (B, H, S, D) float32 CUDA tensor
    """
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # scores: (B, H, S, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn   = F.softmax(scores, dim=-1)
    out    = torch.matmul(attn, v)
    return out
