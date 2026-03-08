/**
 * ThunderKittens implementation of Scaled Dot-Product Multihead Attention
 *   Out = softmax(Q * K^T / sqrt(D)) * V
 *
 * Targets sm_75 (Turing / RTX 2080 Ti).
 *
 * Algorithm (Flash Attention forward, tiled with online softmax)
 * -------------------------------------------------------------
 *   Tile size: BLOCK_S × BLOCK_D  (query/key/value sequence × head dim)
 *   Grid: (ceil(S/BLOCK_S), H, B)
 *   Block: 1 warp (32 threads)
 *
 *   For each query tile q_tile:
 *     Load Q tile [BLOCK_S, BLOCK_D] into shared memory → registers
 *     Init: m = -inf, l = 0, acc = 0
 *
 *     For each kv_tile:
 *       1. Load K tile [BLOCK_S, BLOCK_D] → shared
 *       2. Compute S = Q @ K^T  [BLOCK_S, BLOCK_S] * scale  (scalar GEMM)
 *       3. Compute rowmax m_new; alpha = exp(m - m_new) per row
 *       4. P = exp(S - m_new)             [BLOCK_S, BLOCK_S]
 *       5. Load V tile [BLOCK_S, BLOCK_D] → shared
 *       6. acc = alpha * acc + P @ V      (accumulate weighted sum)
 *       7. l   = alpha * l + rowsum(P)
 *       8. m   = m_new
 *
 *     Out = acc / l  (normalize)
 *     Store Out tile to HBM
 *
 * Notes
 * -----
 * • sm_75 lacks Ampere's `mma.sync.m16n8k16.bf16` so we use fp32 scalar GEMM
 *   for both QK^T and PV products.
 * • TK tile infrastructure is used for HBM→SMEM transfers (warp::load / store),
 *   shared allocator, and tile types. The inner GEMM uses per-thread scalar MACs
 *   (same pattern as matmul_tk.cu).
 * • BLOCK_S must be ≤ 32 for the PV product to fit in registers.
 *   We use BLOCK_S = 32, BLOCK_D = 64.
 *
 * TK types used
 * -------------
 *   gl<float, -1, -1, -1, -1>           – general 4-D global layout (B,H,S,D)
 *   st_fl<BLOCK_S, BLOCK_D>             – shared tile for Q/K/V/Out
 *   shared_allocator                    – dynamic shared memory manager
 */

#include <algorithm>  // std::copy_n
#include "kittens.cuh"
using namespace kittens;

static constexpr int MHA_BLOCK_S  = 32;   // sequence tile dimension
static constexpr int MHA_BLOCK_D  = 64;   // head-dim tile dimension (must ≡ BLOCK_D)
static constexpr int NUM_THREADS   = WARP_THREADS;  // 32

struct mha_globals {
    // Q, K, V, Out: (B, H, S, D) row-major
    float* Q;
    float* K;
    float* V;
    float* Out;

    int B, H, S, D;
    float scale;

    // Strides
    int stride_B, stride_H, stride_S, stride_D;
};

__global__ void mha_fwd_kernel(const __grid_constant__ mha_globals g) {
    const int q_tile = blockIdx.x;  // query tile index
    const int head   = blockIdx.y;
    const int batch  = blockIdx.z;
    const int lane   = kittens::laneid();

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // Two tiles: one for Q (kept), one for K/V (reused per kv_tile)
    auto &Q_s = al.allocate<st_fl<MHA_BLOCK_S, MHA_BLOCK_D>>();
    auto &KV_s = al.allocate<st_fl<MHA_BLOCK_S, MHA_BLOCK_D>>();

    const int base_offset = (batch * g.stride_B + head * g.stride_H);
    const int q_start     = q_tile * MHA_BLOCK_S;
    const int N_OH_OW     = g.S;  // alias for clarity

    // ── Load Q tile into shared and broadcast ──────────────────────────────
    // Use TK gl to load: Q[batch, head, q_start:q_start+BLOCK_S, 0:BLOCK_D]
    // We'll use raw pointer loads since Q is a 4-D tensor without a sub_tile param
    // (TK gl<float,B,H,-1,-1> needs all leading extents known at runtime)
    //
    // Manual load: 32 threads × 2 elements per thread = 64 floats per D row
    // For BLOCK_S=32 rows × BLOCK_D=64 cols = 2048 floats → each thread: 64 floats
    {
        float* q_ptr = g.Q + base_offset + q_start * g.stride_S;
        // 32 threads load 32×64 = 2048 floats: thread t loads row t (BLOCK_S=32=WARP_THREADS)
        // The shared tile st_fl uses a swizzled layout; access via operator{row, col}
        for (int d = 0; d < MHA_BLOCK_D; d += NUM_THREADS) {
            int d_idx = d + lane;
            if (d_idx < MHA_BLOCK_D) {
                // Load BLOCK_S rows, d_idx-th column
                for (int row = 0; row < MHA_BLOCK_S; row++) {
                    int s_abs = q_start + row;
                    Q_s[{row, d_idx}] = (s_abs < g.S && d_idx < g.D)
                        ? q_ptr[row * g.stride_S + d_idx * g.stride_D]
                        : 0.0f;
                }
            }
        }
        __syncthreads();
    }

    // Q register: each thread reads its column from Q_s
    // thread lane reads column `lane` of Q (for BLOCK_D=64, lane range 0..31: 2 cols each)
    // We'll store 32 rows × 2 cols = 64 floats per thread
    float Q_reg[MHA_BLOCK_S * 2];  // Q_reg[row*2 + col_sub]  col = lane*2 + col_sub
    #pragma unroll
    for (int row = 0; row < MHA_BLOCK_S; row++) {
        Q_reg[row * 2    ] = Q_s[{row, lane * 2    }];
        Q_reg[row * 2 + 1] = Q_s[{row, lane * 2 + 1}];
    }

    // Online softmax accumulators: per row (BLOCK_S rows)
    float m[MHA_BLOCK_S];  // row max
    float l[MHA_BLOCK_S];  // row sum of exp
    float acc[MHA_BLOCK_S * MHA_BLOCK_D];  // output accumulator [BLOCK_S, BLOCK_D]
    // Thread t owns acc[row, t*2] and acc[row, t*2+1] for all rows
    #pragma unroll
    for (int i = 0; i < MHA_BLOCK_S; i++) {
        m[i] = -1e30f;
        l[i] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < MHA_BLOCK_S * 2; i++) acc[i] = 0.0f;

    const int num_kv_tiles = (g.S + MHA_BLOCK_S - 1) / MHA_BLOCK_S;

    // ── Flash Attention loop over K/V tiles ──────────────────────────────────
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * MHA_BLOCK_S;
        float* k_ptr = g.K + base_offset + kv_start * g.stride_S;

        // Load K tile into KV_s
        #pragma unroll
        for (int d = 0; d < MHA_BLOCK_D; d += NUM_THREADS) {
            int d_idx = d + lane;
            if (d_idx < MHA_BLOCK_D) {
                for (int row = 0; row < MHA_BLOCK_S; row++) {
                    int s_abs = kv_start + row;
                    KV_s[{row, d_idx}] = (s_abs < g.S && d_idx < g.D)
                        ? k_ptr[row * g.stride_S + d_idx * g.stride_D]
                        : 0.0f;
                }
            }
        }
        __syncthreads();

        // ── Compute S = Q @ K^T * scale  [BLOCK_S, BLOCK_S] ──────────────────
        // Each thread computes one column of S (S[:, lane]) using Q and K
        // S[row, lane] = sum_d Q[row, d] * K[lane, d]
        // Equivalently: each thread owns lane-th kv row; compute inner product with all Q rows
        float S_col[MHA_BLOCK_S];  // S_col[row] = S[row, lane]
        #pragma unroll
        for (int row = 0; row < MHA_BLOCK_S; row++) {
            float dot = 0.0f;
            #pragma unroll
            for (int d_sub = 0; d_sub < 2; d_sub++) {
                // Our thread owns Q columns lane*2 and lane*2+1
                // And K row `lane`, columns lane*2 and lane*2+1... NO.
                // Actually for QK^T we need full inner product over D.
                // We need to reduce over all D columns, not just our 2.
                // Use KV_s shared tile: lane reads KV_s[lane, d] for all d.
                (void)d_sub;
            }
            // Full inner product Q[row, :] · K[lane, :] using shared tile KV_s
            dot = 0.0f;
            for (int d_idx = 0; d_idx < MHA_BLOCK_D; d_idx++) {
                // Q[row, d_idx] needs to come from Q_s (shared) since Q_reg only has our 2 cols
                dot += Q_s[{row, d_idx}] * KV_s[{lane, d_idx}];
            }
            S_col[row] = dot * g.scale;
            // Mask out-of-bounds kv positions
            if (kv_start + lane >= g.S) S_col[row] = -1e30f;
        }

        // ── Online softmax update ─────────────────────────────────────────────
        // Each thread holds one column of S. We need rowmax and rowsum across all 32 threads.
        // Step 1: find new row max via warp reduce
        float m_new[MHA_BLOCK_S];
        #pragma unroll
        for (int row = 0; row < MHA_BLOCK_S; row++) {
            float val = S_col[row];
            // Warp-level max reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
            }
            m_new[row] = fmaxf(m[row], val);
        }

        // Step 2: P = exp(S - m_new), rowsum of P via warp reduce
        float P_col[MHA_BLOCK_S];
        float l_new[MHA_BLOCK_S];
        float alpha[MHA_BLOCK_S];
        #pragma unroll
        for (int row = 0; row < MHA_BLOCK_S; row++) {
            alpha[row]  = expf(m[row] - m_new[row]);
            P_col[row]  = expf(S_col[row] - m_new[row]);
            // Mask OOB kv positions
            if (kv_start + lane >= g.S) P_col[row] = 0.0f;
        }
        // rowsum of P via warp reduce
        #pragma unroll
        for (int row = 0; row < MHA_BLOCK_S; row++) {
            float val = P_col[row];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                val += __shfl_xor_sync(0xffffffff, val, offset);
            }
            l_new[row] = alpha[row] * l[row] + val;
        }

        // Step 3: acc = alpha * acc  (correction for old max)
        #pragma unroll
        for (int row = 0; row < MHA_BLOCK_S; row++) {
            acc[row * 2    ] *= alpha[row];
            acc[row * 2 + 1] *= alpha[row];
        }

        // Step 4: acc += P @ V  (write P to shared, load V, compute P@V)
        // Write P column (P_col) back to KV_s for P@V computation
        // KV_s is now repurposed to store P[BLOCK_S, BLOCK_S]
        // Lane `lane` owns column `lane` of P: P[row, lane] = P_col[row]
        #pragma unroll
        for (int row = 0; row < MHA_BLOCK_S; row++) {
            KV_s[{row, lane}] = P_col[row];
        }
        // Fill rest of columns (lane=32..63 out of range: BLOCK_D=64 but only BLOCK_S=32 cols needed)
        if (lane + NUM_THREADS < MHA_BLOCK_D) {
            #pragma unroll
            for (int row = 0; row < MHA_BLOCK_S; row++) {
                KV_s[{row, lane + NUM_THREADS}] = 0.0f;
            }
        }
        __syncthreads();

        // Load V tile
        float* v_ptr = g.V + base_offset + kv_start * g.stride_S;
        // Reuse Q_s for V (Q_reg holds Q values so Q_s can be clobbered)
        #pragma unroll
        for (int d = 0; d < MHA_BLOCK_D; d += NUM_THREADS) {
            int d_idx = d + lane;
            if (d_idx < MHA_BLOCK_D) {
                for (int row = 0; row < MHA_BLOCK_S; row++) {
                    int s_abs = kv_start + row;
                    Q_s[{row, d_idx}] = (s_abs < g.S && d_idx < g.D)
                        ? v_ptr[row * g.stride_S + d_idx * g.stride_D]
                        : 0.0f;
                }
            }
        }
        __syncthreads();

        // P @ V: thread lane computes output cols lane*2 and lane*2+1
        // (P @ V)[row, col] = sum_j P[row, j] * V[j, col]
        //                     j = 0..BLOCK_S-1 (positions in this kv_tile)
        #pragma unroll
        for (int row = 0; row < MHA_BLOCK_S; row++) {
            float pv0 = 0.0f, pv1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < MHA_BLOCK_S; j++) {
                float p = KV_s[{row, j}];
                pv0 += p * Q_s[{j, lane * 2    }];
                pv1 += p * Q_s[{j, lane * 2 + 1}];
            }
            acc[row * 2    ] += pv0;
            acc[row * 2 + 1] += pv1;
        }

        __syncthreads();

        // Update m and l
        #pragma unroll
        for (int row = 0; row < MHA_BLOCK_S; row++) {
            m[row] = m_new[row];
            l[row] = l_new[row];
        }

        // Restore Q_s from Q_reg (needed for next iteration's QK^T computation)
        #pragma unroll
        for (int row = 0; row < MHA_BLOCK_S; row++) {
            Q_s[{row, lane * 2    }] = Q_reg[row * 2    ];
            Q_s[{row, lane * 2 + 1}] = Q_reg[row * 2 + 1];
        }
        __syncthreads();
    }

    // ── Normalize and store output ─────────────────────────────────────────────
    float* out_ptr = g.Out + base_offset + q_start * g.stride_S;
    #pragma unroll
    for (int row = 0; row < MHA_BLOCK_S; row++) {
        int s_abs = q_start + row;
        if (s_abs >= g.S) continue;

        float inv_l = (l[row] > 0.0f) ? (1.0f / l[row]) : 0.0f;

        // Thread lane writes columns lane*2 and lane*2+1
        #pragma unroll
        for (int col_sub = 0; col_sub < 2; col_sub++) {
            int d_abs = lane * 2 + col_sub;
            if (d_abs < g.D) {
                out_ptr[row * g.stride_S + d_abs * g.stride_D] =
                    acc[row * 2 + col_sub] * inv_l;
            }
        }
    }
}

// ─── PyTorch / Python binding ─────────────────────────────────────────────────
#ifdef TORCH_COMPILE
#include <torch/extension.h>
#include <cmath>

torch::Tensor tk_mha_fwd(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    double scale_d
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Inputs must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(Q.dim() == 4, "Q must be (B, H, S, D)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q, K, V must have the same shape");

    const int B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);

    TORCH_CHECK(D <= MHA_BLOCK_D,
                "Head dim D must be ≤ MHA_BLOCK_D (", MHA_BLOCK_D, "), got D=", D);

    float scale = (scale_d <= 0.0) ? (1.0f / sqrtf((float)D)) : (float)scale_d;

    torch::Tensor Out = torch::empty_like(Q);

    mha_globals g;
    g.Q   = (float*)Q.data_ptr();
    g.K   = (float*)K.data_ptr();
    g.V   = (float*)V.data_ptr();
    g.Out = (float*)Out.data_ptr();
    g.B = B; g.H = H; g.S = S; g.D = D;
    g.scale = scale;
    // Strides (Q, K, V, Out all share same layout)
    g.stride_B = Q.stride(0);
    g.stride_H = Q.stride(1);
    g.stride_S = Q.stride(2);
    g.stride_D = Q.stride(3);

    int q_tiles = (S + MHA_BLOCK_S - 1) / MHA_BLOCK_S;
    dim3 grid(q_tiles, H, B);

    // Shared: Q_s + KV_s = 2 × st_fl<BLOCK_S, BLOCK_D>
    size_t smem = 2 * sizeof(st_fl<MHA_BLOCK_S, MHA_BLOCK_D>);
    cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    mha_fwd_kernel<<<grid, NUM_THREADS, smem>>>(g);
    cudaDeviceSynchronize();

    return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_fwd", &tk_mha_fwd,
          "ThunderKittens Multihead Attention forward (sm_75): Out = softmax(QK^T/sqrt(D)) * V. "
          "Q, K, V: (B, H, S, D) fp32 CUDA, D ≤ 64.",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("scale") = -1.0);
}
#endif  // TORCH_COMPILE
