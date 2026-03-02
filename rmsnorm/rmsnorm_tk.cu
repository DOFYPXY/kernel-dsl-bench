/**
 * ThunderKittens implementation of RMSNorm:
 *   y = (x / sqrt(mean(x^2) + eps)) * weight
 * Normalization is applied over the last (hidden) dimension.
 *
 * Targets sm_75 (Turing / RTX 2080 Ti).
 *
 * Algorithm (two-pass, 1 block per 16 batch rows)
 * ------------------------------------------------
 *   Tile  : st_fl<TILE_ROWS=16, TILE_COLS=64>
 *   Grid  : (ceil(batch / TILE_ROWS),)
 *   Block : 1 warp (32 threads)
 *
 *   Pass 1 – accumulate per-row sum of squares
 *     for each column tile t:
 *       load x_tile  ← g.x [ batch_tile, t ]
 *       x_sq         = x_tile * x_tile            (element-wise)
 *       rms_vec     += row_sum(x_sq)               (per-row accumulation)
 *   rms_vec = rsqrt(rms_vec / hidden + eps)        (via warp::apply lambda)
 *
 *   Pass 2 – normalize and scale
 *     for each column tile t:
 *       load x_tile  ← g.x    [ batch_tile, t ]
 *       load w_sv    ← g.weight[ 0, t ]            (sv_fl<TILE_COLS>)
 *       load w_rv    ← w_sv                        (rv <row_vec> of rt_fl)
 *       y_tile       = mul_row(x_tile, rms_vec)    (x * inv_rms, per row)
 *       y_tile       = mul_col(y_tile, w_rv)       (scale by weight, per col)
 *       store g.y    ← y_tile  [ batch_tile, t ]
 *
 * TK types used
 * -------------
 *   gl<float, 1, 1, -1, -1>          – global layout for x and y (batch × hidden)
 *   gl<float, 1, 1, 1,  -1>          – global layout for weight  (1 × hidden)
 *   st_fl<TILE_ROWS, TILE_COLS>       – shared tile for activations
 *   sv_fl<TILE_COLS>                  – shared vector for weight slice
 *   rt_fl<TILE_ROWS, TILE_COLS>       – register tile for compute
 *   rt_fl<TILE_ROWS, TILE_COLS>::col_vec  – per-row accumulator (rms)
 *   rt_fl<TILE_ROWS, TILE_COLS>::row_vec  – per-col weight vector
 */

#include <algorithm>          // std::copy_n needed by TK's LaunchConfig
#include "kittens.cuh"
using namespace kittens;

static constexpr int TILE_ROWS = 16;
static constexpr int TILE_COLS = 64;
static constexpr int NUM_THREADS = WARP_THREADS;  // 32

struct rmsnorm_globals {
    using x_gl = gl<float, 1, 1, -1, -1>;   // (batch, hidden)
    using w_gl = gl<float, 1, 1,  1, -1>;   // (1, hidden)
    x_gl x;
    w_gl weight;
    x_gl y;
    float eps;
    int batch, hidden;
};

__global__ void rmsnorm_kernel(const __grid_constant__ rmsnorm_globals g) {
    int batch_tile = blockIdx.x;  // processes rows [batch_tile*16, batch_tile*16+16)

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto &x_s = al.allocate<st_fl<TILE_ROWS, TILE_COLS>>();
    auto &w_s = al.allocate<sv_fl<TILE_COLS>>();

    using tile_t    = rt_fl<TILE_ROWS, TILE_COLS>;
    using col_vec_t = tile_t::col_vec;  // length TILE_ROWS, one value per row
    using row_vec_t = tile_t::row_vec;  // length TILE_COLS, one value per col

    tile_t    x_reg, x_sq, y_reg;
    col_vec_t rms_vec;
    row_vec_t w_rv;

    int num_col_tiles = (g.hidden + TILE_COLS - 1) / TILE_COLS;
    float hidden_f    = float(g.hidden);
    float eps_f       = g.eps;

    // ── Pass 1: accumulate sum(x^2) per row ──────────────────────────────────
    kittens::warp::zero(rms_vec);
    for (int t = 0; t < num_col_tiles; t++) {
        kittens::warp::load(x_s, g.x, {0, 0, batch_tile, t});
        __syncthreads();
        kittens::warp::load(x_reg, x_s);

        kittens::warp::mul(x_sq, x_reg, x_reg);          // x_sq = x²
        kittens::warp::row_sum(rms_vec, x_sq, rms_vec);  // rms_vec += row_sum(x_sq)
    }

    // Compute inv_rms per row: rms_vec[i] = rsqrt(rms_vec[i] / hidden + eps)
    kittens::warp::apply(rms_vec, rms_vec,
        [hidden_f, eps_f] __device__ (int /*idx*/, float val) -> float {
            return rsqrtf(val / hidden_f + eps_f);
        }
    );

    // ── Pass 2: normalize and scale ──────────────────────────────────────────
    for (int t = 0; t < num_col_tiles; t++) {
        // Reload x for this tile
        kittens::warp::load(x_s, g.x, {0, 0, batch_tile, t});
        __syncthreads();
        kittens::warp::load(x_reg, x_s);

        // Load weight slice [t*TILE_COLS, (t+1)*TILE_COLS) into shared vector
        kittens::warp::load(w_s, g.weight, {0, 0, 0, t});
        __syncthreads();
        // Load shared vector into register row-vector
        kittens::warp::load(w_rv, w_s);

        // y = x * inv_rms  (broadcast inv_rms per row)
        kittens::warp::mul_row(y_reg, x_reg, rms_vec);

        // y = y * weight  (broadcast weight per column)
        kittens::warp::mul_col(y_reg, y_reg, w_rv);

        // Store to global
        kittens::warp::store(g.y, y_reg, {0, 0, batch_tile, t});
    }
}

// ─── PyTorch / Python binding ────────────────────────────────────────────────
#ifdef TORCH_COMPILE
#include <torch/extension.h>

torch::Tensor tk_rmsnorm(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    float eps)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(),
                "x and weight must be on CUDA");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
                "x and weight must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32,
                "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32,
                "weight must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D (batch, hidden)");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1),
                "weight must be 1-D with length equal to x.size(1)");

    const int batch  = x.size(0);
    const int hidden = x.size(1);

    // Pad batch to multiple of TILE_ROWS and hidden to multiple of TILE_COLS
    auto pad = [](int v, int m) { return ((v + m - 1) / m) * m; };
    const int batch_p  = pad(batch,  TILE_ROWS);
    const int hidden_p = pad(hidden, TILE_COLS);

    // Pad x
    torch::Tensor x_p = (batch_p > batch || hidden_p > hidden)
        ? torch::zeros({batch_p, hidden_p}, x.options())
        : x;
    if (batch_p > batch || hidden_p > hidden) {
        x_p.narrow(0, 0, batch).narrow(1, 0, hidden).copy_(x);
    }

    // Pad weight
    torch::Tensor w_p = (hidden_p > hidden)
        ? torch::cat({weight, torch::zeros({hidden_p - hidden}, weight.options())})
        : weight;
    // Shape for gl: (1, hidden_p)
    w_p = w_p.view({1, hidden_p});

    torch::Tensor y_p = torch::empty_like(x_p);

    using x_gl = rmsnorm_globals::x_gl;
    using w_gl = rmsnorm_globals::w_gl;

    x_gl x_gl_arg{(float*)x_p.data_ptr(), nullptr, nullptr, (size_t)batch_p, (size_t)hidden_p};
    w_gl w_gl_arg{(float*)w_p.data_ptr(), nullptr, nullptr, nullptr,         (size_t)hidden_p};
    x_gl y_gl_arg{(float*)y_p.data_ptr(), nullptr, nullptr, (size_t)batch_p, (size_t)hidden_p};

    rmsnorm_globals g{x_gl_arg, w_gl_arg, y_gl_arg, eps, batch_p, hidden_p};

    int batch_tiles = batch_p / TILE_ROWS;
    size_t smem = sizeof(st_fl<TILE_ROWS, TILE_COLS>)
                + sizeof(sv_fl<TILE_COLS>);

    cudaFuncSetAttribute(
        rmsnorm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem
    );
    rmsnorm_kernel<<<batch_tiles, NUM_THREADS, smem>>>(g);

    // Strip padding and return (batch, hidden) result
    return y_p.narrow(0, 0, batch).narrow(1, 0, hidden).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &tk_rmsnorm,
          "ThunderKittens RMSNorm (sm_75): y = x / rms(x) * weight. "
          "x must be fp32, 2-D (batch, hidden), CUDA.");
}
#endif  // TORCH_COMPILE
