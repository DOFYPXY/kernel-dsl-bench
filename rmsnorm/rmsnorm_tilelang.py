"""
TileLang v0.1.0 implementation of RMSNorm.

Uses the naive_splitk_gemv reduction pattern:
  - 2D thread block: (threads_per_row=1, reduce_threads=threads)
  - Thread dim 0 (tn): row index within block — only 1 row per block here
  - Thread dim 1 (tk): reduction lane across hidden dim
  - T.atomic_add into shared memory; compiler inserts __syncthreads__ automatically
  - Thread tk==0 writes rms back; all threads read it after sync
"""

import os
import torch
import tilelang
from tilelang import Profiler
import tilelang.language as T

# Add CUDA to PATH if needed (use standard symlink, not version-specific)
cuda_bin = "/usr/local/cuda/bin"
if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")

_kernel_cache: dict = {}


def tilelang_rmsnorm_kernel(
    batch: int,
    hidden: int,
    eps: float,
    reduce_threads: int = 256,
    dtype: str = "float16",
):
    """
    Grid  : (batch,)
    Block : threads = (1, reduce_threads)
      tn = T.get_thread_binding(0)  → always 0, identifies the row
      tk = T.get_thread_binding(1)  → 0..reduce_threads-1, reduction lane
    """
    tile_k = (hidden + reduce_threads - 1) // reduce_threads  # elements per thread

    @T.prim_func
    def main(
        x: T.Buffer((batch, hidden), dtype),   # type: ignore[valid-type]
        w: T.Buffer((hidden,), dtype),          # type: ignore[valid-type]
        y: T.Buffer((batch, hidden), dtype),    # type: ignore[valid-type]
    ):
        with T.Kernel(batch, threads=(1, reduce_threads)) as row:
            tn = T.get_thread_binding(0)   # always 0
            tk = T.get_thread_binding(1)   # 0..reduce_threads-1

            # Thread-private local storage for this thread's slice
            x_local = T.alloc_local((tile_k,), "float32")
            w_local = T.alloc_local((tile_k,), "float32")

            # Shared scalars: [0] = sq-sum accumulator, [1] = rms value
            smem = T.alloc_shared((2,), "float32")

            # Thread 0 initialises accumulator
            if tk == 0:
                smem[0] = T.float32(0)

            # Accumulate partial sum of squares into local register
            C_accum = T.alloc_local((1,), "float32")
            T.clear(C_accum)

            for i in T.serial(tile_k):
                col = tk * tile_k + i
                if col < hidden:
                    val = x[row, col].astype("float32")
                    x_local[i] = val
                    w_local[i] = w[col].astype("float32")
                    C_accum[0] += val * val
                else:
                    x_local[i] = T.float32(0)
                    w_local[i] = T.float32(0)

            # All threads atomically add their partial sum → compiler adds sync
            T.atomic_add(smem[0], C_accum[0])

            # Now smem[0] = sum(x^2); thread 0 computes rms
            # (compiler-inserted __syncthreads__ guarantees visibility)
            if tk == 0:
                smem[1] = T.sqrt(smem[0] / T.float32(hidden) + T.float32(eps))

            # Second pass: re-read x from global memory
            # (aligned with TK rmsnorm which re-issues warp::load in pass 2)
            for i in T.serial(tile_k):
                col = tk * tile_k + i
                if col < hidden:
                    x_val = x[row, col].astype("float32")   # re-read from HBM
                    y[row, col] = (x_val / smem[1] * w_local[i]).astype(dtype)

    return main


def tilelang_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    assert x.is_contiguous() and weight.is_contiguous()
    assert x.ndim == 2
    assert x.shape[-1] == weight.numel()

    batch, hidden = x.shape

    _dtype_map = {
        torch.float16: "float16",
        torch.float32: "float32",
    }
    dtype_str = _dtype_map.get(x.dtype, "float16")

    reduce_threads = 256
    cache_key = (batch, hidden, eps, reduce_threads, dtype_str)

    if cache_key not in _kernel_cache:
        program = tilelang_rmsnorm_kernel(batch, hidden, eps, reduce_threads, dtype_str)
        rt_mod, params = tilelang.lower(program)
        _kernel_cache[cache_key] = Profiler(rt_mod, params, result_idx=[2])

    return _kernel_cache[cache_key](x, weight)