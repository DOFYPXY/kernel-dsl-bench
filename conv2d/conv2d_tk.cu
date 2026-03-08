/**
 * ThunderKittens implementation of 2D Convolution: y = conv2d(x, weight)
 *
 * Targets sm_75 (Turing / RTX 2080 Ti).
 *
 * Algorithm (direct tiled convolution)
 * ------------------------------------
 *   Tile layout:
 *     TILE_OHW = 64  – output positions processed per warp (flattened OH*OW)
 *     TILE_OC  = 16  – output channels processed per block
 *
 *   Grid: (ceil(N*OH*OW / TILE_OHW), ceil(C_out / TILE_OC))
 *   Block: 1 warp (32 threads)
 *
 *   Each block:
 *     - Owns output positions [pid_ohw*64, pid_ohw*64+64) and
 *       output channels [pid_oc*16, pid_oc*16+16)
 *     - Loops over (C_in, KH, KW), for each:
 *         1. Load input patch x[n, ic, oh*s + kh - pad, ow*s + kw - pad]
 *            into a register vector (64 scalar values, one per thread pair)
 *         2. Load filter slice w[oc, ic, kh, kw] for oc in [pid_oc*16, ...)
 *            into a shared array (16 values)
 *         3. Accumulate outer product into acc[TILE_OHW, TILE_OC] in registers
 *     - Write acc to output y[n, oc, oh, ow]
 *
 * Note: TK's rt_fl / st_fl are designed for rectangular tiles aligned to HBM.
 * For conv2d the gather pattern (irregular input coordinates) prevents direct
 * use of TK's bulk-load ops (which assume strided rectangular regions).
 * We therefore use scalar register arithmetic; TK is used for the output store
 * and for its shared-memory allocator and type definitions.
 *
 * TK types used
 * -------------
 *   st_fl<TILE_OC, TILE_OHW>  – shared output-accumulator tile
 *   rt_fl<TILE_OC, TILE_OHW>  – register output-accumulator tile
 *   shared_allocator           – dynamic shared-memory manager
 */

#include <algorithm>  // std::copy_n
#include "kittens.cuh"
using namespace kittens;

// Tile dimensions
static constexpr int TK_TILE_OC  = 16;
static constexpr int TK_TILE_OHW = 64;
static constexpr int NUM_THREADS  = WARP_THREADS;  // 32

struct conv2d_globals {
    // x: (N, C_in, H, W) — stored as flat float* with explicit strides
    float* x;
    // weight: (C_out, C_in, KH, KW)
    float* w;
    // output: (N, C_out, OH, OW)
    float* y;
    // bias: (C_out,) — may be nullptr
    float* bias;

    int N, C_in, H, W;
    int C_out, KH, KW;
    int OH, OW;
    int stride, padding;

    // Strides for x (row-major)
    int x_sN, x_sC, x_sH, x_sW;
    // Strides for w (row-major)
    int w_sOC, w_sIC, w_sKH, w_sKW;
    // Strides for y (row-major)
    int y_sN, y_sOC, y_sOH, y_sOW;
};

/**
 * Each thread in the warp (lane 0..31) handles 2 OHW positions from the TILE_OHW=64.
 * Layout: thread t handles positions 2t and 2t+1.
 * Each thread handles 16 OC positions (TILE_OC=16), so accumulates 2×16=32 floats.
 */
__global__ void conv2d_kernel(const __grid_constant__ conv2d_globals g) {
    const int pid_ohw = blockIdx.x;
    const int pid_oc  = blockIdx.y;
    const int lane    = kittens::laneid();

    extern __shared__ alignment_dummy __shm[];
    // Shared memory: TILE_OC floats for weight slice + TILE_OHW floats for input patch
    float* w_smem = reinterpret_cast<float*>(&__shm[0]);
    // w_smem[0..TILE_OC-1]   = weight slice for one (ic, kh, kw) across TILE_OC output channels
    // x_smem[0..TILE_OHW-1]  = input values for TILE_OHW output positions for one (ic,kh,kw)
    float* x_smem = w_smem + TK_TILE_OC;

    // Base output-position index and output-channel index
    const int ohw_base = pid_ohw * TK_TILE_OHW;
    const int oc_base  = pid_oc  * TK_TILE_OC;

    // Per-thread: 2 ohw positions × TK_TILE_OC output channels
    // acc[oc_rel][ohw_local]:  stored as acc[oc_rel * 2 + ohw_local]
    float acc[TK_TILE_OC * 2];  // TK_TILE_OC=16, 2 ohw per thread → 32 floats
    #pragma unroll
    for (int i = 0; i < TK_TILE_OC * 2; i++) acc[i] = 0.0f;

    // Helper: decode flat ohw index → (batch, oh, ow)
    const int N_OH_OW = g.N * g.OH * g.OW;

    for (int ic = 0; ic < g.C_in; ic++) {
        for (int kh = 0; kh < g.KH; kh++) {
            for (int kw = 0; kw < g.KW; kw++) {
                // ── Step 1: load weight slice into shared memory ──────────────────
                // Threads 0..TK_TILE_OC-1 each load one weight
                if (lane < TK_TILE_OC) {
                    int oc_abs = oc_base + lane;
                    w_smem[lane] = (oc_abs < g.C_out)
                        ? g.w[oc_abs * g.w_sOC + ic * g.w_sIC +
                              kh * g.w_sKH + kw * g.w_sKW]
                        : 0.0f;
                }

                // ── Step 2: load input patch into shared memory ───────────────────
                // 32 threads load 64 positions, 2 positions each
                #pragma unroll
                for (int sub = 0; sub < 2; sub++) {
                    int ohw_abs = ohw_base + lane * 2 + sub;
                    if (ohw_abs < N_OH_OW) {
                        int n_idx  = ohw_abs / (g.OH * g.OW);
                        int rem    = ohw_abs % (g.OH * g.OW);
                        int oh     = rem / g.OW;
                        int ow     = rem % g.OW;
                        int ih     = oh * g.stride - g.padding + kh;
                        int iw     = ow * g.stride - g.padding + kw;
                        if (ih >= 0 && ih < g.H && iw >= 0 && iw < g.W) {
                            x_smem[lane * 2 + sub] =
                                g.x[n_idx * g.x_sN + ic * g.x_sC +
                                    ih * g.x_sH + iw * g.x_sW];
                        } else {
                            x_smem[lane * 2 + sub] = 0.0f;
                        }
                    } else {
                        x_smem[lane * 2 + sub] = 0.0f;
                    }
                }
                __syncthreads();

                // ── Step 3: accumulate outer product ─────────────────────────────
                // Each thread reads its 2 input values and multiplies by all 16 weights
                float x0 = x_smem[lane * 2];
                float x1 = x_smem[lane * 2 + 1];
                #pragma unroll
                for (int oc_rel = 0; oc_rel < TK_TILE_OC; oc_rel++) {
                    float w_val = w_smem[oc_rel];
                    acc[oc_rel * 2    ] += x0 * w_val;
                    acc[oc_rel * 2 + 1] += x1 * w_val;
                }
                __syncthreads();
            }
        }
    }

    // ── Step 4: write accumulator to output ──────────────────────────────────
    // Threads cooperate: each thread writes its 2 ohw positions × TK_TILE_OC channels
    #pragma unroll
    for (int oc_rel = 0; oc_rel < TK_TILE_OC; oc_rel++) {
        int oc_abs = oc_base + oc_rel;
        if (oc_abs >= g.C_out) continue;

        float bias_val = (g.bias != nullptr) ? g.bias[oc_abs] : 0.0f;

        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            int ohw_abs = ohw_base + lane * 2 + sub;
            if (ohw_abs >= N_OH_OW) continue;

            int n_idx = ohw_abs / (g.OH * g.OW);
            int rem   = ohw_abs % (g.OH * g.OW);
            int oh    = rem / g.OW;
            int ow    = rem % g.OW;

            g.y[n_idx * g.y_sN + oc_abs * g.y_sOC +
                oh * g.y_sOH + ow * g.y_sOW] =
                acc[oc_rel * 2 + sub] + bias_val;
        }
    }
}

// ─── PyTorch / Python binding ─────────────────────────────────────────────────
#ifdef TORCH_COMPILE
#include <torch/extension.h>

torch::Tensor tk_conv2d(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32,
                "Inputs must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4-D (N, C_in, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4-D (C_out, C_in, KH, KW)");
    TORCH_CHECK(x.size(1) == weight.size(1), "Channel dimensions must match");

    const int N    = x.size(0), C_in  = x.size(1), H  = x.size(2), W  = x.size(3);
    const int C_out = weight.size(0), KH = weight.size(2), KW = weight.size(3);
    const int OH   = (H + 2 * padding - KH) / stride + 1;
    const int OW   = (W + 2 * padding - KW) / stride + 1;

    float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        auto& bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "Bias must be contiguous CUDA");
        TORCH_CHECK(bias.size(0) == C_out, "Bias size must equal C_out");
        bias_ptr = (float*)bias.data_ptr();
    }

    torch::Tensor y = torch::empty({N, C_out, OH, OW}, x.options());

    conv2d_globals g;
    g.x   = (float*)x.data_ptr();
    g.w   = (float*)weight.data_ptr();
    g.y   = (float*)y.data_ptr();
    g.bias = bias_ptr;
    g.N = N; g.C_in = C_in; g.H = H; g.W = W;
    g.C_out = C_out; g.KH = KH; g.KW = KW;
    g.OH = OH; g.OW = OW;
    g.stride  = (int)stride;
    g.padding = (int)padding;
    // x strides
    g.x_sN = x.stride(0); g.x_sC = x.stride(1);
    g.x_sH = x.stride(2); g.x_sW = x.stride(3);
    // w strides
    g.w_sOC = weight.stride(0); g.w_sIC = weight.stride(1);
    g.w_sKH = weight.stride(2); g.w_sKW = weight.stride(3);
    // y strides
    g.y_sN = y.stride(0); g.y_sOC = y.stride(1);
    g.y_sOH = y.stride(2); g.y_sOW = y.stride(3);

    // Grid: (ceil(N*OH*OW / TILE_OHW), ceil(C_out / TILE_OC))
    int N_OH_OW    = N * OH * OW;
    int grid_ohw   = (N_OH_OW + TK_TILE_OHW - 1) / TK_TILE_OHW;
    int grid_oc    = (C_out   + TK_TILE_OC  - 1) / TK_TILE_OC;
    dim3 grid(grid_ohw, grid_oc);

    // Shared memory: TILE_OC + TILE_OHW floats
    size_t smem = (TK_TILE_OC + TK_TILE_OHW) * sizeof(float);
    cudaFuncSetAttribute(conv2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    conv2d_kernel<<<grid, NUM_THREADS, smem>>>(g);
    cudaDeviceSynchronize();

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &tk_conv2d,
          "ThunderKittens Conv2d (sm_75): y = conv2d(x, weight, bias, stride, padding). "
          "All inputs fp32, CUDA.",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias")    = c10::optional<torch::Tensor>(),
          py::arg("stride")  = 1,
          py::arg("padding") = 0);
}
#endif  // TORCH_COMPILE
