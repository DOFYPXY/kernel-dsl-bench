"""
Triton implementation of RMSNorm.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
Normalization is applied over the last dimension.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def triton_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one row (one batch element).
    """

    row_id = tl.program_id(axis=0)

    # Offsets for this row
    offsets = row_id * hidden_size + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_size

    # Load row (cast to fp32 for stable accumulation)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    # Compute mean of squares
    rms = tl.sum(x * x, axis=0) / hidden_size
    rms = tl.sqrt(rms + eps)

    # Normalize
    x_norm = x / rms

    # Load weight (broadcast)
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask).to(tl.float32)

    y = x_norm * w

    # Store back (cast to original dtype automatically)
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Triton wrapper for RMSNorm.

    Args:
        x: (batch, hidden)
        weight: (hidden,)
        eps: epsilon for numerical stability

    Returns:
        Same shape tensor as x
    """
    assert x.is_cuda
    assert weight.is_cuda
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert x.shape[-1] == weight.numel()

    batch, hidden = x.shape

    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(hidden)

    grid = (batch,)

    triton_rmsnorm_kernel[grid](
        x,
        weight,
        y,
        hidden,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y