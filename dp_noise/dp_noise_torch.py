"""
PyTorch reference implementation of epsilon-Differential Privacy (Laplace mechanism).

    y = clip(x, clip_norm) + Laplace(0, clip_norm / epsilon)
"""

import torch


def torch_dp_noise(
    x: torch.Tensor,
    epsilon: float,
    clip_norm: float,
    seed: int = 42,
) -> torch.Tensor:
    """
    PyTorch baseline DP Laplace mechanism.

    Args:
        x:         Input tensor (any shape, CUDA)
        epsilon:   Privacy budget (ε > 0)
        clip_norm: L2 clipping bound (C > 0)
        seed:      RNG seed for reproducibility

    Returns:
        Tensor of same shape/dtype/device as x with DP noise applied
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert epsilon > 0, "epsilon must be positive"
    assert clip_norm > 0, "clip_norm must be positive"

    # ── Clip ──────────────────────────────────────────────────────────────
    norm = torch.linalg.norm(x.float())
    clip_factor = torch.clamp(clip_norm / (norm + 1e-12), max=1.0)
    x_clipped = x * clip_factor

    # ── Laplace noise ─────────────────────────────────────────────────────
    sensitivity = clip_norm
    b = sensitivity / epsilon

    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)

    # Sample Laplace(0, b) via inverse CDF on uniform samples
    u = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=gen)
    u = u.clamp(1e-7, 1.0 - 1e-7)
    noise = torch.where(
        u < 0.5,
         b * torch.log(2.0 * u),
        -b * torch.log(2.0 * (1.0 - u)),
    )

    return x_clipped + noise
