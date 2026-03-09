"""Multihead attention kernel implementations across different GPU DSLs."""

from .multihead_attention_torch import torch_multihead_attention

__all__ = ["torch_multihead_attention"]