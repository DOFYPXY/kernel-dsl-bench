import math

import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    QK_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)

    pid_z = pid_zh // H
    pid_h = pid_zh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (
        Q
        + pid_z * stride_qz
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )

    # q: [BLOCK_M, HEAD_DIM]
    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < N_CTX),
        other=0.0,
    )

    # Precomputed on host side to avoid constexpr/Python math interaction issues.
    qk_scale = QK_SCALE

    # running stats for online softmax
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # loop over key/value sequence blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        k_ptrs = (
            K
            + pid_z * stride_kz
            + pid_h * stride_kh
            + (start_n + offs_n)[None, :] * stride_kn
            + offs_d[:, None] * stride_kk
        )
        v_ptrs = (
            V
            + pid_z * stride_vz
            + pid_h * stride_vh
            + (start_n + offs_n)[:, None] * stride_vn
            + offs_d[None, :] * stride_vk
        )

        # k: [HEAD_DIM, BLOCK_N]
        k = tl.load(
            k_ptrs,
            mask=((start_n + offs_n)[None, :] < N_CTX),
            other=0.0,
        )

        # qk: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * qk_scale

        # causal mask
        if CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # mask out rows beyond N_CTX
        row_mask = offs_m[:, None] < N_CTX
        col_mask = (start_n + offs_n[None, :]) < N_CTX
        qk = tl.where(row_mask & col_mask, qk, float("-inf"))

        # online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # v: [BLOCK_N, HEAD_DIM]
        v = tl.load(
            v_ptrs,
            mask=((start_n + offs_n)[:, None] < N_CTX),
            other=0.0,
        )

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # normalize
    acc = acc / l_i[:, None]

    o_ptrs = (
        O
        + pid_z * stride_oz
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_ok
    )

    tl.store(
        o_ptrs,
        acc,
        mask=(offs_m[:, None] < N_CTX),
    )


def torch_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Reference implementation for correctness checking if needed.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        s_q = q.shape[-2]
        s_k = k.shape[-2]
        mask = torch.ones((s_q, s_k), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def triton_multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Triton forward multi-head attention.

    Args:
        q: [B, H, S, D]
        k: [B, H, S, D]
        v: [B, H, S, D]
        causal: whether to apply causal masking

    Returns:
        o: [B, H, S, D]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be CUDA tensors"
    assert q.dtype == k.dtype == v.dtype, "q/k/v dtypes must match"
    assert q.shape == k.shape == v.shape, "q/k/v must have the same shape"
    assert q.ndim == 4, "expected q/k/v shape [B, H, S, D]"

    B, H, S, D = q.shape
    assert D in (16, 32, 64, 128), f"HEAD_DIM={D} not supported yet"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), (
        "please pass contiguous q/k/v"
    )

    o = torch.empty_like(q)

    # Keep tile sizes conservative so the kernel fits 64KB shared-memory GPUs
    # like RTX 2080 Ti (sm_75).
    BLOCK_M = 32
    BLOCK_N = 32
    qk_scale = 1.0 / math.sqrt(D)

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    num_warps = 4 if D <= 64 else 8

    _mha_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, S,
        HEAD_DIM=D,
        QK_SCALE=qk_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
        num_warps=num_warps,
        num_stages=1,
    )

    return o