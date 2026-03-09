"""TileLang implementation of Matrix Multiplication (MatMul) kernel: C = A @ B"""

import os
import torch
import tilelang
from tilelang import Profiler
import tilelang.language as T

<<<<<<< HEAD
os.environ["PATH"] = "/usr/local/cuda-12.1/bin:" + os.environ.get("PATH", "")
=======
# Add CUDA to PATH if needed (use standard symlink, not version-specific)
cuda_bin = "/usr/local/cuda/bin"
if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
>>>>>>> origin/main

# Module-level cache: (M, N, K, block_M, block_N, block_K) -> compiled kernel
_kernel_cache: dict = {}


def tilelang_matmul_kernel(M, N, K, block_M=128, block_N=128, block_K=32,
                            dtype="float16", accum_dtype="float"):
    """
    TileLang kernel: C = A @ B

    Args:
        M, N, K:    Matrix dimensions
        block_M/N/K: Tile sizes for tiling
        dtype:      Input/output data type
        accum_dtype: Accumulator data type
    """
    @T.prim_func
    def main(  # type: ignore[reportArgumentType]
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A from global to shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                # Copy tile of B from global to shared memory
                T.copy(B[k * block_K, bx * block_N], B_shared)
                # Tile-level matrix multiply accumulate
                T.gemm(A_shared, B_shared, C_local)

            # Store result tile from registers to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def tilelang_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    TileLang implementation wrapper for matrix multiplication.

    Args:
        a: Input matrix A of shape (M, K) (must be on CUDA and contiguous)
        b: Input matrix B of shape (K, N) (must be on CUDA and contiguous)

    Returns:
        Output matrix C of shape (M, N) where C = A @ B
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda, "Matrices must be on CUDA device"
    assert a.is_contiguous() and b.is_contiguous(), "Matrices must be contiguous"

    M, K = a.shape
    K, N = b.shape

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    cache_key = (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    if cache_key not in _kernel_cache:
        program = tilelang_matmul_kernel(M, N, K,
                                          block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K)
        rt_mod, params = tilelang.lower(program)
        _kernel_cache[cache_key] = Profiler(rt_mod, params, result_idx=[2])

    kernel = _kernel_cache[cache_key]

    # result_idx=[2] means C is the output: Profiler allocates it and returns it
    return kernel(a, b)