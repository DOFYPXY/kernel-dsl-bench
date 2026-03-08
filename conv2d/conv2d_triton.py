"""Triton implementation of 2D convolution (im2col + GEMM strategy)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv2d_kernel(
    # Input pointers
    x_ptr,
    w_ptr,
    y_ptr,
    # Dimensions
    N, C_in, H, W,
    C_out, KH, KW,
    OH, OW,
    stride,
    padding,
    # Strides for x: (N, C_in, H, W) row-major
    x_sN, x_sC, x_sH, x_sW,
    # Strides for w: (C_out, C_in, KH, KW) row-major
    w_sOC, w_sIC, w_sKH, w_sKW,
    # Strides for y: (N, C_out, OH, OW) row-major
    y_sN, y_sOC, y_sOH, y_sOW,
    # Tile sizes (compile-time constants)
    BLOCK_OC: tl.constexpr,
    BLOCK_OHW: tl.constexpr,
):
    """
    Each program processes BLOCK_OC output channels × BLOCK_OHW output positions.
    Grid: (ceil(N*OH*OW / BLOCK_OHW), ceil(C_out / BLOCK_OC))
    """
    pid_ohw = tl.program_id(0)
    pid_oc  = tl.program_id(1)

    # Output position indices [pid_ohw * BLOCK_OHW, ...) across flattened (N, OH, OW)
    ohw_start = pid_ohw * BLOCK_OHW
    ohw_idx   = ohw_start + tl.arange(0, BLOCK_OHW)  # shape [BLOCK_OHW]
    ohw_mask  = ohw_idx < N * OH * OW

    # Decode flattened index to (n, oh, ow)
    n_idx  = ohw_idx // (OH * OW)
    rem    = ohw_idx % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    # Output channel indices
    oc_start = pid_oc * BLOCK_OC
    oc_idx   = oc_start + tl.arange(0, BLOCK_OC)   # shape [BLOCK_OC]
    oc_mask  = oc_idx < C_out

    # Accumulator: [BLOCK_OHW, BLOCK_OC]
    acc = tl.zeros([BLOCK_OHW, BLOCK_OC], dtype=tl.float32)

    # Loop over (C_in, KH, KW)
    for ic in range(C_in):
        for kh in range(KH):
            for kw in range(KW):
                # Input coordinates: ih = oh * stride - padding + kh
                ih = oh_idx * stride - padding + kh  # [BLOCK_OHW]
                iw = ow_idx * stride - padding + kw  # [BLOCK_OHW]
                valid_hw = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                # Load input values: x[n, ic, ih, iw]
                x_off = (n_idx * x_sN + ic * x_sC +
                         ih * x_sH + iw * x_sW)
                x_val = tl.load(x_ptr + x_off,
                                 mask=ohw_mask & valid_hw,
                                 other=0.0)  # [BLOCK_OHW]

                # Load weight values: w[oc, ic, kh, kw]
                w_off = (oc_idx * w_sOC + ic * w_sIC +
                         kh * w_sKH + kw * w_sKW)
                w_val = tl.load(w_ptr + w_off,
                                 mask=oc_mask,
                                 other=0.0)  # [BLOCK_OC]

                # Outer product accumulate: acc[ohw, oc] += x[ohw] * w[oc]
                acc += x_val[:, None] * w_val[None, :]

    # Write output: y[n, oc, oh, ow]
    out_off = (n_idx[:, None] * y_sN +
               oc_idx[None, :] * y_sOC +
               oh_idx[:, None] * y_sOH +
               ow_idx[:, None] * y_sOW)
    tl.store(y_ptr + out_off,
             acc,
             mask=ohw_mask[:, None] & oc_mask[None, :])


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    2D convolution via Triton (im2col-free, direct outer-product accumulation).

    Args:
        x:      (N, C_in, H, W) float32 CUDA tensor
        weight: (C_out, C_in, KH, KW) float32 CUDA tensor
        bias:   (C_out,) float32 CUDA tensor, optional
        stride: stride for height and width
        padding: zero-padding for height and width

    Returns:
        (N, C_out, OH, OW) float32 CUDA tensor
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    assert x.is_contiguous() and weight.is_contiguous(), "Inputs must be contiguous"
    assert x.dtype == torch.float32 and weight.dtype == torch.float32

    N, C_in, H, W     = x.shape
    C_out, _, KH, KW  = weight.shape

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1

    y = torch.empty((N, C_out, OH, OW), dtype=torch.float32, device=x.device)

    BLOCK_OC  = 16
    BLOCK_OHW = 64

    grid = (
        triton.cdiv(N * OH * OW, BLOCK_OHW),
        triton.cdiv(C_out, BLOCK_OC),
    )

    _conv2d_kernel[grid](
        x, weight, y,
        N, C_in, H, W,
        C_out, KH, KW,
        OH, OW,
        stride, padding,
        # x strides
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        # w strides
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        # y strides
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_OC=BLOCK_OC,
        BLOCK_OHW=BLOCK_OHW,
    )

    if bias is not None:
        y = y + bias.view(1, C_out, 1, 1)

    return y
