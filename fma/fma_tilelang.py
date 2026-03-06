"""TileLang implementation of Fused Multiply-Add (FMA) kernel: y = x * a + b"""

"""
The new TileLang version is incompatible with the RTX 2080 Ti (sm_75, Turing). 
Downgrade TileLang version to 0.1.0.
"""

import os
import torch
import tilelang
from tilelang import Profiler
import tilelang.language as T

os.environ["PATH"] = "/usr/local/cuda-12.1/bin:" + os.environ.get("PATH", "")

# Module-level cache: (N, a, b) -> compiled kernel
_kernel_cache: dict = {}


def tilelang_fma_kernel(N, a, b, num_per_thread=4, threads=256, dtype="float32"):
    """
    TileLang kernel: y = x * a + b (element-wise fused multiply-add).

    Args:
        N:              Number of elements to process
        a:              Scalar multiplier (captured as compile-time constant)
        b:              Scalar addend    (captured as compile-time constant)
        num_per_thread: Elements each thread handles (controls vectorization width)
        threads:        Number of threads per block (compile-time constant)
        dtype:          Data type string, e.g. "float32" or "float16"
    """
    block_size = threads * num_per_thread

    @T.prim_func
    def main(  # type: ignore[reportArgumentType]
        y: T.Buffer((N,), dtype),
        x: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=threads) as bx:
            x_reg = T.alloc_fragment((block_size,), dtype)
            y_reg = T.alloc_fragment((block_size,), dtype)

            s_start = bx * block_size
            s_end   = (bx + 1) * block_size

            # Load x tile from global memory into registers (LDG.128)
            T.copy(x[s_start:s_end], x_reg)

            # Compute y = x * a + b  (a, b captured from outer Python scope)
            for tid, i in T.Parallel(threads, num_per_thread):
                idx = tid * num_per_thread + i
                y_reg[idx] = x_reg[idx] * a + b

            # Store y tile from registers back to global memory (STG.128)
            T.copy(y_reg, y[s_start:s_end])

    return main


def tilelang_fma(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    TileLang implementation wrapper for FMA.

    Args:
        x: Input tensor (must be on CUDA and contiguous)
        a: Scalar multiplier
        b: Scalar addend

    Returns:
        Output tensor y = x * a + b
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.is_contiguous(), "Input must be contiguous"

    N = x.numel()
    NUM_PER_THREAD = 4  # float32: 4×32bit = 128-bit vector load
    THREADS = 256

    cache_key = (N, a, b, NUM_PER_THREAD, THREADS)
    if cache_key not in _kernel_cache:
        program = tilelang_fma_kernel(N, a, b, num_per_thread=NUM_PER_THREAD, threads=THREADS)
        rt_mod, params = tilelang.lower(program)
        _kernel_cache[cache_key] = Profiler(rt_mod, params, result_idx=[0])

    kernel = _kernel_cache[cache_key]

    # result_idx=[0] means y is the output: Profiler allocates it and returns it
    return kernel(x)