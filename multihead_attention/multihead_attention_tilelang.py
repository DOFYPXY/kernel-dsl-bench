#!/usr/bin/env python3
"""
Multihead Attention via TileLang FlashAttention kernel.

Implements FlashAttention-2 online softmax algorithm.

NOTE: TileLang v0.1.0's LayoutInference has limited 3D buffer support.
We flatten (BH, S, D) → (BH*S, D) and use 2D buffers inside the kernel.

NOTE: T.gemm requires fp16 inputs on sm_75 (T4).

Designed for TileLang v0.1.0 on sm_75 (T4 / RTX 2080 Ti).
"""

import math
import torch
import tilelang
import os
import tilelang.language as T

_kernel_cache: dict = {}
cuda_bin = "/usr/local/cuda/bin"
if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")

def _make_flash_attn_program(
    batch_heads,
    seq_len,
    head_dim,
    block_M,
    block_N,
    is_causal=False,
    dtype="float16",
    accum_dtype="float",
):
    """
    Build a TileLang FlashAttention kernel.

    Buffers are 2D: Q_flat/K_flat/V_flat/O_flat all (batch_heads * seq_len, head_dim).
    Grid: (ceildiv(seq_len, block_M), batch_heads)
    """
    scale = 1.0 / math.sqrt(head_dim)
    num_kv_blocks = (seq_len + block_N - 1) // block_N
    total_rows = batch_heads * seq_len

    @T.prim_func
    def flash_attn_kernel(
        Q_flat: T.Buffer((total_rows, head_dim), dtype),
        K_flat: T.Buffer((total_rows, head_dim), dtype),
        V_flat: T.Buffer((total_rows, head_dim), dtype),
        O_flat: T.Buffer((total_rows, head_dim), dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), batch_heads, threads=128) as (bx, bz):
            # ---- Shared memory buffers (all 2D) ----
            Q_shared = T.alloc_shared((block_M, head_dim), dtype)
            K_shared = T.alloc_shared((block_N, head_dim), dtype)
            V_shared = T.alloc_shared((block_N, head_dim), dtype)
            S_shared = T.alloc_shared((block_M, block_N), dtype)
            S_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            O_local  = T.alloc_fragment((block_M, head_dim), accum_dtype)

            # Online softmax state
            m_prev = T.alloc_fragment((block_M,), accum_dtype)
            m_new  = T.alloc_fragment((block_M,), accum_dtype)
            l_prev = T.alloc_fragment((block_M,), accum_dtype)
            l_new  = T.alloc_fragment((block_M,), accum_dtype)

            # Base row offset for this batch-head
            base = bz * seq_len

            # ---- Initialise ----
            T.clear(O_local)
            for i in T.Parallel(block_M):
                m_prev[i] = -1e4
                l_prev[i] = 0.0

            # Load Q tile (reused across all K/V blocks)
            T.copy(Q_flat[base + bx * block_M, 0], Q_shared)

            # ---- Iterate over K/V blocks ----
            for j in T.serial(num_kv_blocks):

                # Load K and V tiles
                T.copy(K_flat[base + j * block_N, 0], K_shared)

                # S = Q @ K^T
                T.clear(S_local)
                T.gemm(Q_shared, K_shared, S_local, transpose_B=True)

                # Scale
                for mi, ni in T.Parallel(block_M, block_N):
                    S_local[mi, ni] *= scale

                # Causal mask
                if is_causal:
                    for mi, ni in T.Parallel(block_M, block_N):
                        if bx * block_M + mi < j * block_N + ni:
                            S_local[mi, ni] = -1e4

                # ---- Online softmax ----
                # Row max
                T.reduce_max(S_local, m_new, dim=1)
                for i in T.Parallel(block_M):
                    m_new[i] = T.max(m_new[i], m_prev[i])

                # Rescale O and l
                for i, d in T.Parallel(block_M, head_dim):
                    O_local[i, d] *= T.exp(m_prev[i] - m_new[i])
                for i in T.Parallel(block_M):
                    l_prev[i] *= T.exp(m_prev[i] - m_new[i])

                # P = exp(S - m_new)
                for mi, ni in T.Parallel(block_M, block_N):
                    S_local[mi, ni] = T.exp(S_local[mi, ni] - m_new[mi])

                # Row sum
                T.reduce_sum(S_local, l_new, dim=1)
                for i in T.Parallel(block_M):
                    l_prev[i] += l_new[i]

                # Load V tile
                T.copy(V_flat[base + j * block_N, 0], V_shared)

                # O += P @ V  (S_local fp32 → S_shared fp16 for T.gemm)
                T.copy(S_local, S_shared)
                T.gemm(S_shared, V_shared, O_local)

                # Advance max
                for i in T.Parallel(block_M):
                    m_prev[i] = m_new[i]

            # ---- Final normalisation ----
            for i, d in T.Parallel(block_M, head_dim):
                O_local[i, d] /= l_prev[i]

            # Write output
            T.copy(O_local, O_flat[base + bx * block_M, 0])

    return flash_attn_kernel


def _compile_flash_attn(batch_heads, seq_len, head_dim, is_causal):
    """Compile (and cache) a TileLang FlashAttention kernel."""
    key = (batch_heads, seq_len, head_dim, is_causal)
    if key in _kernel_cache:
        return _kernel_cache[key]

    block_M = 64
    block_N = 64

    program = _make_flash_attn_program(
        batch_heads, seq_len, head_dim,
        block_M, block_N,
        is_causal=is_causal,
        dtype="float16",
        accum_dtype="float",
    )

    # v0.1.0: out_idx=[3] → 4th buffer (O_flat) is the output
    mod, params = tilelang.lower(program)
    kernel = tilelang.Profiler(mod, params, [3], tilelang.TensorSupplyType.Integer)

    _kernel_cache[key] = kernel
    return kernel


def tilelang_multihead_attention(q, k, v, causal=False):
    """
    Multihead attention using TileLang FlashAttention kernel.

    Parameters
    ----------
    q, k, v : Tensor (B, H, S, D)
    causal  : bool

    Returns
    -------
    Tensor (B, H, S, D)
    """
    B, H, S, D = q.shape
    orig_dtype = q.dtype
    BH = B * H

    # Flatten to 2D: (BH * S, D) — avoids 3D buffer issues in v0.1.0
    q_fp16 = q.reshape(BH * S, D).contiguous()
    k_fp16 = k.reshape(BH * S, D).contiguous()
    v_fp16 = v.reshape(BH * S, D).contiguous()

    kernel = _compile_flash_attn(BH, S, D, causal)
    output = kernel(q_fp16, k_fp16, v_fp16)

    output = output.reshape(B, H, S, D)
    if orig_dtype != torch.float16:
        output = output.to(orig_dtype)

    return output