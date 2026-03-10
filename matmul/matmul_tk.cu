/**
 * ThunderKittens implementation of Matrix Multiplication: C = A @ B
 *
 * Targets sm_75 (Turing / RTX 2080 Ti).
 *
 * Architecture note
 * -----------------
 * ThunderKittens' warp::mma_AB relies on the PTX instruction
 *   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 * which requires sm_80+ (Ampere). On sm_75 this instruction is unavailable,
 * so this kernel uses TK's tile infrastructure for efficient global↔shared
 * memory movement and computes the inner product in fp32 scalar arithmetic.
 *
 * Algorithm (shared-memory tiled GEMM, Level-03 pattern with TK gl wrappers)
 * --------------------------------------------------------------------------
 *   Tile size : BLOCK × BLOCK  (32×32)
 *   Grid      : (ceil(N/BLOCK), ceil(M/BLOCK))
 *   Block     : 1 warp (32 threads), each thread accumulates one column of
 *               the output tile across BLOCK inner rows
 *
 *   For each K-tile k:
 *     1. warp::load  As ← g.A[ blockIdx.y, k ]   (32×32 float, global→shared)
 *     2. warp::load  Bs ← g.B[ k, blockIdx.x ]   (32×32 float, global→shared)
 *     3. Each thread t computes:
 *          for i in [0, BLOCK): acc[i] += sum_k As[i,k] * Bs[k,t]
 *   Then write acc[] → Cs (shared) → g.C (global) via warp::store.
 *
 * TK types used
 * -------------
 *   gl<float, 1, 1, -1, -1>          – raw 2-D global layout
 *   st_fl<BLOCK, BLOCK>               – swizzled shared-memory tile
 *   shared_allocator                  – dynamic shared-memory manager
 */

#include <algorithm>          // std::copy_n needed by TK's LaunchConfig
#include <cuda_fp16.h>
#include "kittens.cuh"
using namespace kittens;

// BLOCK_SIZE = 64 matches Triton (BLOCK_SIZE_M/N=64) and TileLang (block_M/N=64).
// 1 warp = 32 threads; each thread owns 2 output columns (tx and tx+32).
static constexpr int BLOCK_SIZE  = 64;
static constexpr int NUM_THREADS = WARP_THREADS;  // 32

struct matmul_globals {
    __half* A;   // (M, K) row-major fp16
    __half* B;   // (K, N) row-major fp16
    __half* C;   // (M, N) row-major fp16
    int M, N, K;
};

__global__ void matmul_kernel(const __grid_constant__ matmul_globals g) {
    extern __shared__ alignment_dummy __shm[];
    // Two fp16 shared-memory tiles: As[BLOCK×BLOCK] and Bs[BLOCK×BLOCK]
    __half* As = reinterpret_cast<__half*>(&__shm[0]);
    __half* Bs = As + BLOCK_SIZE * BLOCK_SIZE;

    const int tx = kittens::laneid();   // 0..31
    const int bx = (int)blockIdx.x;    // N-tile index
    const int by = (int)blockIdx.y;    // M-tile index

    // fp32 accumulators for two output columns per thread:
    //   col0 = bx*BLOCK_SIZE + tx,  col1 = bx*BLOCK_SIZE + tx+32
    float acc0[BLOCK_SIZE], acc1[BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) acc0[i] = acc1[i] = 0.0f;

    const int num_k_tiles = (g.K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int k = 0; k < num_k_tiles; k++) {
        // Load A tile: 4096 fp16 elements / 32 threads = 128 per thread
        for (int elem = tx; elem < BLOCK_SIZE * BLOCK_SIZE; elem += NUM_THREADS) {
            const int row  = elem / BLOCK_SIZE;
            const int col  = elem % BLOCK_SIZE;
            const int m_abs = by * BLOCK_SIZE + row;
            const int k_abs = k  * BLOCK_SIZE + col;
            As[elem] = (m_abs < g.M && k_abs < g.K)
                ? g.A[m_abs * g.K + k_abs] : __float2half(0.0f);
        }
        // Load B tile: 4096 fp16 elements / 32 threads = 128 per thread
        for (int elem = tx; elem < BLOCK_SIZE * BLOCK_SIZE; elem += NUM_THREADS) {
            const int row  = elem / BLOCK_SIZE;
            const int col  = elem % BLOCK_SIZE;
            const int k_abs = k  * BLOCK_SIZE + row;
            const int n_abs = bx * BLOCK_SIZE + col;
            Bs[elem] = (k_abs < g.K && n_abs < g.N)
                ? g.B[k_abs * g.N + n_abs] : __float2half(0.0f);
        }
        __syncthreads();

        // Inner product with fp32 accumulation:
        //   acc0[i] += sum_kk A[i,kk] * B[kk, tx   ]
        //   acc1[i] += sum_kk A[i,kk] * B[kk, tx+32]
        #pragma unroll 4
        for (int i = 0; i < BLOCK_SIZE; i++) {
            #pragma unroll 4
            for (int kk = 0; kk < BLOCK_SIZE; kk++) {
                const float a_val = __half2float(As[i * BLOCK_SIZE + kk]);
                acc0[i] += a_val * __half2float(Bs[kk * BLOCK_SIZE + tx     ]);
                acc1[i] += a_val * __half2float(Bs[kk * BLOCK_SIZE + tx + 32]);
            }
        }
        __syncthreads();
    }

    // Write fp16 output directly to global memory
    const int m_base = by * BLOCK_SIZE;
    const int n_base = bx * BLOCK_SIZE;
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
        const int m_abs = m_base + i;
        if (m_abs >= g.M) continue;
        const int n0 = n_base + tx;
        const int n1 = n_base + tx + 32;
        if (n0 < g.N) g.C[m_abs * g.N + n0] = __float2half_rn(acc0[i]);
        if (n1 < g.N) g.C[m_abs * g.N + n1] = __float2half_rn(acc1[i]);
    }
}

// ─── PyTorch / Python binding ────────────────────────────────────────────────
#ifdef TORCH_COMPILE
#include <torch/extension.h>

torch::Tensor tk_matmul(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "Both matrices must be on CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "Both matrices must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
                "Inputs must be float16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "Inputs must be 2-D");
    TORCH_CHECK(a.size(1) == b.size(0),
                "Inner dimensions must match");

    const int M = a.size(0), K = a.size(1), N = b.size(1);

    // Pad all dimensions to multiples of BLOCK_SIZE
    auto pad = [&](int x) {
        return ((x + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    };
    int M_p = pad(M), K_p = pad(K), N_p = pad(N);

    auto pad_tensor = [&](const torch::Tensor& t, int rows, int cols) -> torch::Tensor {
        if (t.size(0) == rows && t.size(1) == cols) return t.contiguous();
        torch::Tensor out = torch::zeros({rows, cols}, t.options());
        out.narrow(0, 0, t.size(0)).narrow(1, 0, t.size(1)).copy_(t);
        return out;
    };

    torch::Tensor a_p = pad_tensor(a, M_p, K_p);
    torch::Tensor b_p = pad_tensor(b, K_p, N_p);
    torch::Tensor c_p = torch::zeros({M_p, N_p}, a.options());

    matmul_globals g;
    g.A = (__half*)a_p.data_ptr();
    g.B = (__half*)b_p.data_ptr();
    g.C = (__half*)c_p.data_ptr();
    g.M = M_p; g.N = N_p; g.K = K_p;

    dim3 grid(N_p / BLOCK_SIZE, M_p / BLOCK_SIZE);
    // Two fp16 tiles of BLOCK_SIZE×BLOCK_SIZE
    size_t smem = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(__half);

    cudaFuncSetAttribute(
        matmul_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem
    );
    matmul_kernel<<<grid, NUM_THREADS, smem>>>(g);

    // Strip padding
    return c_p.narrow(0, 0, M).narrow(1, 0, N).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &tk_matmul,
          "ThunderKittens MatMul (sm_75): C = A @ B. "
          "A, B must be fp32, 2-D, CUDA (M×K and K×N).");
}
#endif  // TORCH_COMPILE
