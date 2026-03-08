/**
 * ThunderKittens implementation of epsilon-Differential Privacy via Laplace mechanism.
 *
 *   y = clip(x, clip_norm) + Laplace(0, clip_norm / epsilon)
 *
 * Targets sm_75 (Turing / RTX 2080 Ti).  Uses warp-level TK ops for tile
 * arithmetic and cuRAND Philox for Laplace noise generation on-device.
 *
 * Two-phase algorithm (no host↔device sync between phases)
 * ─────────────────────────────────────────────────────────
 *   Phase 1 – dp_norm_sq_kernel
 *     Each block loads a tile of X, computes thread-local sum-of-squares via
 *     direct register-tile data access, does a warp-level reduction, and
 *     atomicAdds the partial sum to a global scalar `norm_sq`.
 *
 *   Phase 2 – dp_clip_noise_kernel
 *     Reads `norm_sq` from global memory (safe because CUDA streams are
 *     ordered), computes clip_factor and Laplace scale on-device, then for
 *     each tile:
 *       • loads X tile via TK (global → shared → register)
 *       • clips:  x_clipped = x * clip_factor
 *       • generates per-element Laplace noise using cuRAND Philox directly
 *         in the register tile data arrays
 *       • y = x_clipped + noise
 *       • stores y back to global memory
 *
 * Tile layout (same as fma_tk.cu)
 * ────────────────────────────────
 *   sub-tile : st_fl<16, 64>  →  1024 fp32 elements per warp
 *   grid     : (col_tiles=1, row_tiles = N_padded / 1024)
 *   block    : 1 warp (32 threads)
 */

#include <algorithm>
#include "kittens.cuh"
#include <curand_kernel.h>

using namespace kittens;

static constexpr int TILE_ROWS = 16;
static constexpr int TILE_COLS = 64;
static constexpr int TILE_SIZE = TILE_ROWS * TILE_COLS;  // 1024
static constexpr int NUM_THREADS = WARP_THREADS;          // 32

/* ════════════════════════════════════════════════════════════════════════════
 *  Phase 1 – Compute ‖x‖² (sum of squares) via tile-level accumulation
 * ════════════════════════════════════════════════════════════════════════════ */

struct dp_norm_globals {
    using sub_tile = st_fl<TILE_ROWS, TILE_COLS>;
    using gl_t     = gl<float, 1, 1, -1, -1, sub_tile>;
    gl_t x;
    float* norm_sq_ptr;   // device pointer – output scalar
};

__global__ void dp_norm_sq_kernel(const __grid_constant__ dp_norm_globals g) {
    int row = blockIdx.y;
    int col = blockIdx.x;  // always 0 for 1-D input reshaped to 2-D

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto &x_s = al.allocate<st_fl<TILE_ROWS, TILE_COLS>>();

    rt_fl<TILE_ROWS, TILE_COLS> x_reg;

    kittens::warp::load(x_s, g.x, {0, 0, row, col});
    __syncthreads();
    kittens::warp::load(x_reg, x_s);

    // Thread-local sum of x² over owned register-tile elements
    float thread_sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < x_reg.height; i++) {
        #pragma unroll
        for (int j = 0; j < x_reg.width; j++) {
            #pragma unroll
            for (int k = 0; k < x_reg.packed_per_tile; k++) {
                float a = x_reg.tiles[i][j].data[k].x;
                float b = x_reg.tiles[i][j].data[k].y;
                thread_sum_sq += a * a + b * b;
            }
        }
    }

    // Intra-warp reduction (butterfly pattern)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum_sq += __shfl_down_sync(0xFFFFFFFF, thread_sum_sq, offset);

    // Lane 0 accumulates into global scalar
    if (threadIdx.x == 0)
        atomicAdd(g.norm_sq_ptr, thread_sum_sq);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Phase 2 – Clip + Laplace noise addition
 * ════════════════════════════════════════════════════════════════════════════ */

struct dp_clip_noise_globals {
    using sub_tile = st_fl<TILE_ROWS, TILE_COLS>;
    using gl_t     = gl<float, 1, 1, -1, -1, sub_tile>;
    gl_t x, y;
    float* norm_sq_ptr;          // written by phase-1
    float  clip_norm;
    float  epsilon;
    unsigned long long seed;
};

__global__ void dp_clip_noise_kernel(const __grid_constant__ dp_clip_noise_globals g) {
    int row = blockIdx.y;
    int col = blockIdx.x;

    // ── Compute clip_factor and Laplace scale on-device ──────────────────
    float norm_val    = sqrtf(*g.norm_sq_ptr);
    float clip_factor = (norm_val > g.clip_norm)
                        ? (g.clip_norm / norm_val)
                        : 1.0f;
    float laplace_b   = g.clip_norm / g.epsilon;

    // ── Per-thread Philox RNG (unique sequence per thread) ───────────────
    int global_id = (row * gridDim.x + col) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t rng;
    curand_init(g.seed, (unsigned long long)global_id, 0, &rng);

    // ── Load X tile: global → shared → register ─────────────────────────
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto &x_s = al.allocate<st_fl<TILE_ROWS, TILE_COLS>>();

    rt_fl<TILE_ROWS, TILE_COLS> x_reg;

    kittens::warp::load(x_s, g.x, {0, 0, row, col});
    __syncthreads();
    kittens::warp::load(x_reg, x_s);

    // ── Clip: x_clipped = x * clip_factor ────────────────────────────────
    kittens::warp::mul(x_reg, x_reg, clip_factor);

    // ── Generate Laplace noise and add it directly in registers ──────────
    //
    //  Laplace(0, b) via inverse CDF, branch-free formulation:
    //    U ~ Uniform(0, 1)
    //    if U < 0.5:  noise =  b * ln(2U)
    //    else:        noise = -b * ln(2(1-U))
    //
    //  curand_uniform returns (0, 1], clamped to avoid log(0).

    #pragma unroll
    for (int i = 0; i < x_reg.height; i++) {
        #pragma unroll
        for (int j = 0; j < x_reg.width; j++) {
            #pragma unroll
            for (int k = 0; k < x_reg.packed_per_tile; k++) {
                // Element .x
                {
                    float u = curand_uniform(&rng);
                    u = fminf(fmaxf(u, 1e-7f), 1.0f - 1e-7f);
                    float noise = (u < 0.5f)
                        ?  laplace_b * logf(2.0f * u)
                        : -laplace_b * logf(2.0f * (1.0f - u));
                    x_reg.tiles[i][j].data[k].x += noise;
                }
                // Element .y
                {
                    float u = curand_uniform(&rng);
                    u = fminf(fmaxf(u, 1e-7f), 1.0f - 1e-7f);
                    float noise = (u < 0.5f)
                        ?  laplace_b * logf(2.0f * u)
                        : -laplace_b * logf(2.0f * (1.0f - u));
                    x_reg.tiles[i][j].data[k].y += noise;
                }
            }
        }
    }

    // ── Store result: register → global ──────────────────────────────────
    kittens::warp::store(g.y, x_reg, {0, 0, row, col});
}

/* ════════════════════════════════════════════════════════════════════════════
 *  PyTorch / Python binding
 * ════════════════════════════════════════════════════════════════════════════ */
#ifdef TORCH_COMPILE
#include <torch/extension.h>

torch::Tensor tk_dp_noise(
    const torch::Tensor& x,
    float epsilon,
    float clip_norm,
    int64_t seed)
{
    TORCH_CHECK(x.is_cuda(),       "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(epsilon > 0.0f,    "epsilon must be positive");
    TORCH_CHECK(clip_norm > 0.0f,  "clip_norm must be positive");

    const int N = x.numel();

    // Pad to multiple of TILE_SIZE (1024)
    const int N_padded = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    // Reshape to 2-D: (TOTAL_ROWS, TILE_COLS) so TK tile descriptors work
    torch::Tensor x2d = (N_padded > N)
        ? torch::cat({x.view({-1}),
                      torch::zeros({N_padded - N}, x.options())})
              .view({N_padded / TILE_COLS, TILE_COLS})
        : x.view({-1}).view({N_padded / TILE_COLS, TILE_COLS});

    torch::Tensor y2d = torch::empty_like(x2d);

    const int TOTAL_ROWS = N_padded / TILE_COLS;
    const int ROW_TILES  = TOTAL_ROWS / TILE_ROWS;

    using gl_t = dp_norm_globals::gl_t;
    size_t smem = sizeof(st_fl<TILE_ROWS, TILE_COLS>);

    // Allocate device scalar for norm²  (zero-initialized)
    torch::Tensor norm_sq_t = torch::zeros({1}, x.options());
    float* norm_sq_ptr = (float*)norm_sq_t.data_ptr();

    // ── Phase 1: Compute ‖x‖² ───────────────────────────────────────────
    gl_t x_gl{(float*)x2d.data_ptr(), nullptr, nullptr,
              (size_t)TOTAL_ROWS, (size_t)TILE_COLS};

    dp_norm_globals g_norm{x_gl, norm_sq_ptr};

    dim3 grid(1, ROW_TILES);

    cudaFuncSetAttribute(
        dp_norm_sq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);
    dp_norm_sq_kernel<<<grid, NUM_THREADS, smem>>>(g_norm);

    // ── Phase 2: Clip + Laplace noise ────────────────────────────────────
    //  No host sync needed — same-stream ordering guarantees phase-1
    //  results are visible to phase-2.
    gl_t y_gl{(float*)y2d.data_ptr(), nullptr, nullptr,
              (size_t)TOTAL_ROWS, (size_t)TILE_COLS};

    dp_clip_noise_globals g_clip{
        x_gl, y_gl,
        norm_sq_ptr,
        clip_norm,
        epsilon,
        (unsigned long long)seed
    };

    cudaFuncSetAttribute(
        dp_clip_noise_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);
    dp_clip_noise_kernel<<<grid, NUM_THREADS, smem>>>(g_clip);

    // Flatten and strip padding
    return y2d.view({N_padded}).narrow(0, 0, N).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dp_noise", &tk_dp_noise,
          "ThunderKittens DP Laplace mechanism (sm_75): "
          "y = clip(x, clip_norm) + Laplace(0, clip_norm/epsilon). "
          "x must be fp32, 1-D, CUDA. seed controls the Philox RNG.");
}
#endif  // TORCH_COMPILE
