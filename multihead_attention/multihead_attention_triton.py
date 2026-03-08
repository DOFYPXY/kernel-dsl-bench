"""Triton implementation of scaled dot-product multihead attention (Flash Attention style)."""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Strides: (B, H, S, D) row-major
    stride_qB, stride_qH, stride_qS, stride_qD,
    stride_kB, stride_kH, stride_kS, stride_kD,
    stride_vB, stride_vH, stride_vS, stride_vD,
    stride_oB, stride_oH, stride_oS, stride_oD,
    # Dimensions
    S, D,
    scale,
    # Tile sizes (compile-time)
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_KV_TILES: tl.constexpr,
):
    """
    Flash Attention forward pass.
    Grid: (ceil(S/BLOCK_S), H, B)
    Each program handles BLOCK_S query rows for one (batch, head).
    """
    q_tile = tl.program_id(0)  # query tile index
    head   = tl.program_id(1)
    batch  = tl.program_id(2)

    q_start = q_tile * BLOCK_S

    # Base pointers for this (batch, head) — NOT pre-offset by q_start
    q_base = Q_ptr   + batch * stride_qB + head * stride_qH
    k_base = K_ptr   + batch * stride_kB + head * stride_kH
    v_base = V_ptr   + batch * stride_vB + head * stride_vH
    o_base = Out_ptr + batch * stride_oB + head * stride_oH

    # Row/col ranges
    q_rows = q_start + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    d_cols = tl.arange(0, BLOCK_D)              # [BLOCK_D]
    q_mask = q_rows < S                          # [BLOCK_S]

    # Load Q tile: [BLOCK_S, BLOCK_D]
    Q = tl.load(
        q_base + q_rows[:, None] * stride_qS + d_cols[None, :] * stride_qD,
        mask=q_mask[:, None] & (d_cols[None, :] < D),
        other=0.0,
    )

    # Online softmax accumulators
    m_i  = tl.full([BLOCK_S], float("-inf"), dtype=tl.float32)  # row max
    l_i  = tl.zeros([BLOCK_S], dtype=tl.float32)                # row sum of exp
    acc  = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)       # output accumulator

    # Stream over K/V in blocks
    for kv_tile in range(NUM_KV_TILES):
        kv_start = kv_tile * BLOCK_S
        kv_rows  = kv_start + tl.arange(0, BLOCK_S)
        kv_mask  = kv_rows < S

        # Load K tile: [BLOCK_S, BLOCK_D]
        K = tl.load(
            k_base + kv_rows[:, None] * stride_kS + d_cols[None, :] * stride_kD,
            mask=kv_mask[:, None] & (d_cols[None, :] < D),
            other=0.0,
        )

        # QK^T: [BLOCK_S, BLOCK_S]  (q_rows by kv_rows)
        qk = tl.dot(Q, tl.trans(K), allow_tf32=False) * scale
        # Mask out-of-bounds kv positions
        qk = tl.where(kv_mask[None, :], qk, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha  = tl.exp(m_i - m_new)          # correction factor for old acc
        p      = tl.exp(qk - m_new[:, None])  # [BLOCK_S, BLOCK_S]

        # Load V tile: [BLOCK_S, BLOCK_D]
        V = tl.load(
            v_base + kv_rows[:, None] * stride_vS + d_cols[None, :] * stride_vD,
            mask=kv_mask[:, None] & (d_cols[None, :] < D),
            other=0.0,
        )

        acc  = acc * alpha[:, None] + tl.dot(p, V, allow_tf32=False)
        l_i  = l_i * alpha + tl.sum(p, axis=1)
        m_i  = m_new

    # Normalize
    out = acc / l_i[:, None]

    # Store output: rows q_start..q_start+BLOCK_S-1
    tl.store(
        o_base + q_rows[:, None] * stride_oS + d_cols[None, :] * stride_oD,
        out,
        mask=q_mask[:, None] & (d_cols[None, :] < D),
    )


def triton_multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Triton Flash Attention forward pass.

    Args:
        q: (B, H, S, D) float32 CUDA tensor
        k: (B, H, S, D) float32 CUDA tensor
        v: (B, H, S, D) float32 CUDA tensor
        scale: attention scale (default: 1/sqrt(D))

    Returns:
        (B, H, S, D) float32 CUDA tensor
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.dtype == torch.float32

    B, H, S, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)

    # Choose tile sizes: keep shared memory within 48 KB for sm_75.
    # Shared usage: Q[BLOCK_S, BLOCK_D] + K[BLOCK_S, BLOCK_D] + V[BLOCK_S, BLOCK_D]
    # + P[BLOCK_S, BLOCK_S] in fp32 → 4*(3*BLOCK_S*BLOCK_D + BLOCK_S²) bytes.
    # With BLOCK_D=64 and BLOCK_S=32: 4*(3*32*64 + 32*32) = 4*(6144+1024) = 28672 < 48KB ✓
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_S = 32  # Conservative to stay within sm_75 shared memory limit
    num_kv_tiles = triton.cdiv(S, BLOCK_S)

    grid = (triton.cdiv(S, BLOCK_S), H, B)

    _mha_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S, D,
        scale,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        NUM_KV_TILES=num_kv_tiles,
        num_warps=4,
    )

    return out
