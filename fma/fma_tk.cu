/**
 * ThunderKittens implementation of Fused Multiply-Add (FMA) kernel: y = x * a + b
 *
 * Targets sm_75 (Turing / RTX 2080 Ti). Uses warp-level TK ops for element-wise
 * arithmetic — no tensor-core MMA required. The 1D input is reshaped in Python to
 * a 2D matrix (num_tile_rows × TILE_COLS) so TK tile descriptors can index it.
 *
 * Tile layout
 * -----------
 *   sub-tile : st_fl<TILE_ROWS=16, TILE_COLS=64>  →  1024 fp32 elements per warp
 *   grid     : (col_tiles=1, row_tiles = N_padded / 1024)
 *   block    : 1 warp (32 threads)
 */

#include <algorithm>          // std::copy_n needed by TK's LaunchConfig
#include "kittens.cuh"
using namespace kittens;

static constexpr int TILE_ROWS = 16;
static constexpr int TILE_COLS = 64;
static constexpr int NUM_THREADS = WARP_THREADS;  // 32

struct fma_globals {
    using sub_tile = st_fl<TILE_ROWS, TILE_COLS>;
    using gl_t     = gl<float, 1, 1, -1, -1, sub_tile>;
    gl_t x, y;
    float a, b;
};

__global__ void fma_kernel(const __grid_constant__ fma_globals g) {
    int row = blockIdx.y;  // row tile index
    int col = blockIdx.x;  // col tile index (always 0 for 1D inputs)

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto &x_s = al.allocate<st_fl<TILE_ROWS, TILE_COLS>>();

    rt_fl<TILE_ROWS, TILE_COLS> x_reg, y_reg;

    // Global → Shared → Register
    kittens::warp::load(x_s, g.x, {0, 0, row, col});
    __syncthreads();
    kittens::warp::load(x_reg, x_s);

    // y = x * a + b
    kittens::warp::mul(y_reg, x_reg, g.a);
    kittens::warp::add(y_reg, y_reg, g.b);

    // Register → Global
    kittens::warp::store(g.y, y_reg, {0, 0, row, col});
}

// ─── PyTorch / Python binding ────────────────────────────────────────────────
#ifdef TORCH_COMPILE
#include <torch/extension.h>

torch::Tensor tk_fma(const torch::Tensor& x, float a, float b) {
    TORCH_CHECK(x.is_cuda(),       "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    const int N         = x.numel();
    const int TILE_SIZE = TILE_ROWS * TILE_COLS;  // 1024

    // Pad N to a multiple of TILE_SIZE
    const int N_padded = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    // Reshape to 2D: (N_padded / TILE_COLS, TILE_COLS) so that each st_fl<16,64>
    // tile covers 16 contiguous rows × 64 columns of the 2D view.
    torch::Tensor x2d = (N_padded > N)
        ? torch::cat({x, torch::zeros({N_padded - N}, x.options())})
              .view({N_padded / TILE_COLS, TILE_COLS})
        : x.view({N_padded / TILE_COLS, TILE_COLS});

    torch::Tensor y2d = torch::empty_like(x2d);

    const int TOTAL_ROWS = N_padded / TILE_COLS;   // e.g. 156250 for N=10M
    const int ROW_TILES  = TOTAL_ROWS / TILE_ROWS; // groups of 16 rows

    using gl_t = fma_globals::gl_t;
    gl_t x_gl{(float*)x2d.data_ptr(), nullptr, nullptr, (size_t)TOTAL_ROWS, (size_t)TILE_COLS};
    gl_t y_gl{(float*)y2d.data_ptr(), nullptr, nullptr, (size_t)TOTAL_ROWS, (size_t)TILE_COLS};
    fma_globals g{x_gl, y_gl, a, b};

    dim3 grid(1, ROW_TILES);
    size_t smem = sizeof(st_fl<TILE_ROWS, TILE_COLS>);
    cudaFuncSetAttribute(fma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    fma_kernel<<<grid, NUM_THREADS, smem>>>(g);
    cudaDeviceSynchronize();

    // Flatten and strip padding
    return y2d.view({N_padded}).narrow(0, 0, N).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fma", &tk_fma,
          "ThunderKittens FMA (sm_75): y = x * a + b. "
          "x must be fp32, 1-D, CUDA.");
}
#endif  // TORCH_COMPILE
