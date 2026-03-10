#!/usr/bin/env python3
"""
Conv2D via direct convolution in TileLang (Issue-3 fix: no im2col to global memory).

Algorithm mirrors TK's conv2d_tk.cu:
  - Grid  : (ceildiv(N*H_out*W_out, BLOCK_OHW),  ceildiv(C_out, BLOCK_OC))
  - Block : 32 threads (1 warp), BLOCK_OHW=64, BLOCK_OC=16
  - For each (ic, kh, kw):
      1. Load weight slice w[pid_oc*BLOCK_OC:(pid_oc+1)*BLOCK_OC, ic, kh, kw]
         into shared memory  (Issue-7: explicit smem caching, matching TK)
      2. Load input patch  x[n, ic, ih, iw] for each ohw position
         into shared memory
      3. Accumulate outer-product into register accumulators

Shared memory is used for BOTH the weight slice and the input patch,
matching TK's memory layout exactly.

Designed for TileLang v0.1.0 on sm_75 (T4 / RTX 2080 Ti).
"""
import os
import torch
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Kernel cache – avoids recompilation
# ---------------------------------------------------------------------------
_kernel_cache: dict = {}
cuda_bin = "/usr/local/cuda/bin"
if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")

# Tile dimensions matching TK's conv2d_tk.cu
_BLOCK_OC  = 16   # output channels per block
_BLOCK_OHW = 64   # output spatial positions per block (32 threads × 2 positions)
_THREADS   = 32   # 1 warp per block


def _make_direct_conv_kernel(
    N, C_in, H, W, C_out, H_out, W_out, pad, stride, dtype="float32"
):
    """
    Build a TileLang direct-convolution kernel (3×3, specialised for N=1).

    Memory layout (aligned with TK conv2d_tk.cu):
      w_smem[BLOCK_OC]   – weight slice for one (ic, kh, kw) across BLOCK_OC output channels
      x_smem[BLOCK_OHW]  – input values for BLOCK_OHW output positions for one (ic, kh, kw)

    TileLang's ThreadStorageSync pass inserts __syncthreads__ between
    adjacent T.Parallel blocks that access shared memory, providing the
    necessary barriers between the smem-write and smem-read phases.
    """
    block_oc  = _BLOCK_OC
    block_ohw = _BLOCK_OHW
    threads   = _THREADS

    @T.prim_func
    def main(
        x: T.Buffer((N, C_in, H, W), dtype),
        w: T.Buffer((C_out, C_in, 3, 3), dtype),
        y: T.Buffer((N, C_out, H_out, W_out), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N * H_out * W_out, block_ohw),
            T.ceildiv(C_out, block_oc),
            threads=threads,
        ) as (pid_ohw, pid_oc):

            # ── Shared memory (Issue-7): weight slice + input patch ──────────
            w_smem = T.alloc_shared((block_oc,),  "float32")
            x_smem = T.alloc_shared((block_ohw,), "float32")

            # ── Register accumulator: thread tid owns positions tid*2, tid*2+1
            #    across all block_oc output channels  (32 floats per thread)
            acc = T.alloc_local((block_oc * 2,), "float32")
            T.clear(acc)

            tid = T.get_thread_binding(0)   # 0..31

            total_ohw = N * H_out * W_out

            for ic in T.serial(C_in):
                for kh in T.serial(3):
                    for kw in T.serial(3):

                        # ── Step 1: load weight slice into smem ──────────────
                        # T.Parallel(block_oc) → threads 0..15 each load 1 weight;
                        # threads 16..31 are idle for this step.
                        # TileLang inserts __syncthreads__ after T.Parallel.
                        for oc_local in T.Parallel(block_oc):
                            oc_abs = pid_oc * block_oc + oc_local
                            if oc_abs < C_out:
                                w_smem[oc_local] = w[oc_abs, ic, kh, kw].astype("float32")
                            else:
                                w_smem[oc_local] = T.float32(0)

                        # ── Step 2: load input patch into smem ───────────────
                        # T.Parallel(block_ohw)=64 with 32 threads → 2 loads per thread.
                        # TileLang inserts __syncthreads__ after T.Parallel.
                        for ohw_local in T.Parallel(block_ohw):
                            ohw_abs = pid_ohw * block_ohw + ohw_local
                            if ohw_abs < total_ohw:
                                n_idx = ohw_abs // (H_out * W_out)
                                rem   = ohw_abs - n_idx * (H_out * W_out)
                                oh    = rem // W_out
                                ow    = rem - oh * W_out
                                ih    = oh * stride - pad + kh
                                iw    = ow * stride - pad + kw
                                if (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W):
                                    x_smem[ohw_local] = x[n_idx, ic, ih, iw].astype("float32")
                                else:
                                    x_smem[ohw_local] = T.float32(0)
                            else:
                                x_smem[ohw_local] = T.float32(0)

                        # ── Step 3: accumulate outer product ─────────────────
                        # Thread tid reads its two positions from x_smem and
                        # all BLOCK_OC weights from w_smem (serial → per-thread).
                        for oc_rel in T.serial(block_oc):
                            acc[oc_rel * 2    ] += w_smem[oc_rel] * x_smem[tid * 2    ]
                            acc[oc_rel * 2 + 1] += w_smem[oc_rel] * x_smem[tid * 2 + 1]

            # ── Step 4: write output ─────────────────────────────────────────
            for oc_rel in T.serial(block_oc):
                oc_abs = pid_oc * block_oc + oc_rel
                if oc_abs < C_out:
                    for sub in T.serial(2):
                        ohw_abs = pid_ohw * block_ohw + tid * 2 + sub
                        if ohw_abs < total_ohw:
                            n_idx = ohw_abs // (H_out * W_out)
                            rem   = ohw_abs - n_idx * (H_out * W_out)
                            oh    = rem // W_out
                            ow    = rem - oh * W_out
                            y[n_idx, oc_abs, oh, ow] = acc[oc_rel * 2 + sub].astype(dtype)

    return main


def _compile_direct_conv(N, C_in, H, W, C_out, H_out, W_out, pad, stride, dtype_str):
    key = (N, C_in, H, W, C_out, H_out, W_out, pad, stride, dtype_str)
    if key in _kernel_cache:
        return _kernel_cache[key]

    program = _make_direct_conv_kernel(
        N, C_in, H, W, C_out, H_out, W_out, pad, stride, dtype=dtype_str
    )
    mod, params = tilelang.lower(program)
    # result_idx=[2] → y is the 3rd buffer (output)
    kernel = tilelang.Profiler(mod, params, [2], tilelang.TensorSupplyType.Integer)
    _kernel_cache[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tilelang_conv2d(x, w, bias=None, stride=1, padding=1):
    """
    Direct Conv2D in TileLang (no im2col scratch buffer).
    Shared memory is used for weight slices and input patches, matching TK.

    Parameters
    ----------
    x : Tensor  (N, Cin, H, W)       – input activations
    w : Tensor  (Cout, Cin, KH, KW)  – convolution filters (KH=KW=3)
    bias : Tensor or None
    stride : int
    padding : int

    Returns
    -------
    Tensor  (N, Cout, H_out, W_out)
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    assert x.is_contiguous() and w.is_contiguous(), "Inputs must be contiguous"
    assert w.size(2) == 3 and w.size(3) == 3, "Kernel specialised for 3×3"

    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    dtype_str = dtype_map[x.dtype]

    N_batch, C_in, H, W = x.shape
    C_out, _, KH, KW = w.shape
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    kernel = _compile_direct_conv(
        N_batch, C_in, H, W, C_out, H_out, W_out,
        padding, stride, dtype_str
    )

    output = kernel(x, w)   # returns y of shape (N, C_out, H_out, W_out)

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)

    return output