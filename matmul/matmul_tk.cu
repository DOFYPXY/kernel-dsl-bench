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
#include "kittens.cuh"
using namespace kittens;

static constexpr int BLOCK_SIZE  = 32;
static constexpr int NUM_THREADS = WARP_THREADS;  // 32

struct matmul_globals {
    using gl_t = gl<float, 1, 1, -1, -1>;
    gl_t A, B, C;
    int M, N, K;
};

__global__ void matmul_kernel(const __grid_constant__ matmul_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto &As = al.allocate<st_fl<BLOCK_SIZE, BLOCK_SIZE>>();
    auto &Bs = al.allocate<st_fl<BLOCK_SIZE, BLOCK_SIZE>>();
    auto &Cs = al.allocate<st_fl<BLOCK_SIZE, BLOCK_SIZE>>();

    int tx = kittens::laneid();  // 0..31: thread handles output column tx

    // Each thread accumulates BLOCK_SIZE output values
    float acc[BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) acc[i] = 0.0f;

    int num_k_tiles = (g.K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int k = 0; k < num_k_tiles; k++) {
        // Load A[ blockIdx.y, k ] and B[ k, blockIdx.x ] from HBM → shared
        kittens::warp::load(As, g.A, {0, 0, (int)blockIdx.y, k});
        kittens::warp::load(Bs, g.B, {0, 0, k, (int)blockIdx.x});
        __syncthreads();

        // Inner product: each thread owns one output column (tx)
        //   acc[i] = sum_kk  A[i, kk] * B[kk, tx]
        #pragma unroll 8
        for (int i = 0; i < BLOCK_SIZE; i++) {
            #pragma unroll 8
            for (int kk = 0; kk < BLOCK_SIZE; kk++) {
                acc[i] += As[{i, kk}] * Bs[{kk, tx}];
            }
        }
        __syncthreads();
    }

    // Write per-thread accumulator into shared output tile
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
        Cs[{i, tx}] = acc[i];
    }
    __syncthreads();

    // Shared → Global
    kittens::warp::store(g.C, Cs, {0, 0, (int)blockIdx.y, (int)blockIdx.x});
}

// ─── PyTorch / Python binding ────────────────────────────────────────────────
#ifdef TORCH_COMPILE
#include <torch/extension.h>

torch::Tensor tk_matmul(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "Both matrices must be on CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "Both matrices must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32,
                "Inputs must be float32");
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

    using gl_t = matmul_globals::gl_t;
    gl_t a_gl{(float*)a_p.data_ptr(), nullptr, nullptr, (size_t)M_p, (size_t)K_p};
    gl_t b_gl{(float*)b_p.data_ptr(), nullptr, nullptr, (size_t)K_p, (size_t)N_p};
    gl_t c_gl{(float*)c_p.data_ptr(), nullptr, nullptr, (size_t)M_p, (size_t)N_p};
    matmul_globals g{a_gl, b_gl, c_gl, M_p, N_p, K_p};

    dim3 grid(N_p / BLOCK_SIZE, M_p / BLOCK_SIZE);
    size_t smem = 3 * sizeof(st_fl<BLOCK_SIZE, BLOCK_SIZE>);

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
