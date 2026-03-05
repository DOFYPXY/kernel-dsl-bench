"""
Triton implementation of Conv2D (NCHW), specialized for:
- kernel size: 3x3
- stride: 1
- padding: 1
- groups: 1

y[n, co, h, w] = sum_{ci, kh, kw} x[n, ci, h+kh-1, w+kw-1] * w[co, ci, kh, kw]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def conv2d3x3_s1p1_kernel(
    x_ptr,  # *fp16/fp32
    w_ptr,  # *fp16/fp32
    y_ptr,  # *fp16/fp32
    N: tl.constexpr,
    C_IN: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_OUT: tl.constexpr,
    STRIDE_N: tl.constexpr,
    STRIDE_CI: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    W_STRIDE_CO: tl.constexpr,
    W_STRIDE_CI: tl.constexpr,
    # weight strides for kh,kw are contiguous in last dims
    Y_STRIDE_N: tl.constexpr,
    Y_STRIDE_CO: tl.constexpr,
    Y_STRIDE_H: tl.constexpr,
    Y_STRIDE_W: tl.constexpr,
    BLOCK_HW: tl.constexpr,     # number of output pixels per program
    BLOCK_CO: tl.constexpr,     # number of output channels per program
):
    """
    Program maps a block of output pixels (flattened h*w) and a block of output channels.
    """
    pid_hw = tl.program_id(axis=0)
    pid_co = tl.program_id(axis=1)

    # output spatial size (same as input when s=1,p=1,k=3)
    H_OUT = H
    W_OUT = W
    HW_OUT = H_OUT * W_OUT

    # block of flattened positions
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < HW_OUT

    h = hw_offsets // W_OUT
    w = hw_offsets - h * W_OUT

    # block of output channels
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < C_OUT

    # We benchmark N=1 by default, but keep N for completeness:
    n = 0  # single-batch specialization for simplicity

    # accumulator: [BLOCK_CO, BLOCK_HW]
    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)

    # iterate over input channels and 3x3 kernel
    for ci in range(0, C_IN):
        # base pointers for x at this ci
        # apply padding=1: input indices are (h + kh - 1, w + kw - 1)
        for kh in range(0, 3):
            ih = h + kh - 1
            mask_h = (ih >= 0) & (ih < H)

            for kw in range(0, 3):
                iw = w + kw - 1
                mask_w = (iw >= 0) & (iw < W)

                x_mask = mask_hw & mask_h & mask_w

                x_idx = (
                    n * STRIDE_N
                    + ci * STRIDE_CI
                    + ih * STRIDE_H
                    + iw * STRIDE_W
                )
                x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)  # [BLOCK_HW]

                # load weights for all co in the block at (ci,kh,kw)
                w_idx = (
                    co_offsets[:, None] * W_STRIDE_CO
                    + ci * W_STRIDE_CI
                    + kh * 3
                    + kw
                )
                w_val = tl.load(w_ptr + w_idx, mask=mask_co[:, None], other=0.0)  # [BLOCK_CO,1]

                # fused multiply-add (broadcast x across co)
                acc += w_val.to(tl.float32) * x_val[None, :].to(tl.float32)

    # store
    y_hw = hw_offsets
    y_h = h
    y_w = w
    y_idx = (
        n * Y_STRIDE_N
        + co_offsets[:, None] * Y_STRIDE_CO
        + y_h[None, :] * Y_STRIDE_H
        + y_w[None, :] * Y_STRIDE_W
    )
    tl.store(y_ptr + y_idx, acc.to(tl.float32), mask=mask_co[:, None] & mask_hw[None, :])


def triton_conv2d_3x3_s1p1(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Triton Conv2D wrapper specialized for:
      x: (N, C_in, H, W) with N==1
      w: (C_out, C_in, 3, 3)
      stride=1, padding=1

    Returns:
      y: (1, C_out, H, W)
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    assert x.is_contiguous(), "x must be contiguous (NCHW contiguous)"
    assert w.is_contiguous(), "w must be contiguous"
    assert x.dim() == 4 and w.dim() == 4
    N, C_IN, H, W = x.shape
    C_OUT, C_IN2, KH, KW = w.shape
    assert N == 1, "This simple kernel currently supports N=1 for benchmarking"
    assert C_IN2 == C_IN
    assert KH == 3 and KW == 3, "This kernel is specialized for 3x3"
    y = torch.empty((N, C_OUT, H, W), device=x.device, dtype=torch.float32)

    # strides in elements
    STRIDE_N = x.stride(0)
    STRIDE_CI = x.stride(1)
    STRIDE_H = x.stride(2)
    STRIDE_W = x.stride(3)

    W_STRIDE_CO = w.stride(0)
    W_STRIDE_CI = w.stride(1)

    Y_STRIDE_N = y.stride(0)
    Y_STRIDE_CO = y.stride(1)
    Y_STRIDE_H = y.stride(2)
    Y_STRIDE_W = y.stride(3)

    BLOCK_HW = 256
    BLOCK_CO = 32

    H_OUT = H
    W_OUT = W
    HW_OUT = H_OUT * W_OUT

    grid = (
        triton.cdiv(HW_OUT, BLOCK_HW),
        triton.cdiv(C_OUT, BLOCK_CO),
    )

    conv2d3x3_s1p1_kernel[grid](
        x, w, y,
        N=N, C_IN=C_IN, H=H, W=W, C_OUT=C_OUT,
        STRIDE_N=STRIDE_N, STRIDE_CI=STRIDE_CI, STRIDE_H=STRIDE_H, STRIDE_W=STRIDE_W,
        W_STRIDE_CO=W_STRIDE_CO, W_STRIDE_CI=W_STRIDE_CI,
        Y_STRIDE_N=Y_STRIDE_N, Y_STRIDE_CO=Y_STRIDE_CO, Y_STRIDE_H=Y_STRIDE_H, Y_STRIDE_W=Y_STRIDE_W,
        BLOCK_HW=BLOCK_HW, BLOCK_CO=BLOCK_CO,
        num_warps=4,
    )
    return y