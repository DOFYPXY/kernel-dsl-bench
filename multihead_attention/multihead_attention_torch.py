"""PyTorch implementation of multihead attention."""

import math

import torch
import torch.nn.functional as F


def torch_multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    PyTorch baseline multihead attention.

    Args:
        q: Query tensor of shape (batch, heads, seq_q, head_dim)
        k: Key tensor of shape (batch, heads, seq_k, head_dim)
        v: Value tensor of shape (batch, heads, seq_k, head_dim)
        causal: Whether to apply a causal mask

    Returns:
        Output tensor of shape (batch, heads, seq_q, head_dim)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be CUDA tensors"
    assert q.shape[:-1] == k.shape[:-1] == v.shape[:-1], "q/k/v shapes must align"
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "head dimensions must match"

    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # Fallback for older PyTorch versions without SDPA.
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_q = q.shape[-2]
        seq_k = k.shape[-2]
        mask = torch.ones((seq_q, seq_k), device=q.device, dtype=torch.bool).triu(1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, v)