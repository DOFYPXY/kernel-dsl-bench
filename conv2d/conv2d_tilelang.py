#!/usr/bin/env python3
"""
Conv2D via implicit im2col + TileLang GEMM.

Strategy:
  1. Use torch.nn.functional.unfold (im2col) to rearrange input patches
     into a 2D matrix of shape (Cin*KH*KW, H_out*W_out).
  2. Reshape weight to (Cout, Cin*KH*KW).
  3. Run a TileLang GEMM kernel:  Output = Weight @ Im2col_matrix
  4. Reshape result back to (N, Cout, H_out, W_out).

NOTE: TileLang v0.1.0's T.gemm uses CUTLASS tensor core MMA which is
optimized for fp16 inputs. This implementation now supports multiple data types
(float32, float16, bfloat16) by reading the dtype from input tensors.

Designed for TileLang v0.1.0 on sm_75 (T4 / RTX 2080 Ti).
"""
import os
import torch
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Kernel cache – avoids recompilation across repeated benchmark invocations
# ---------------------------------------------------------------------------
_kernel_cache: dict = {}
cuda_bin = "/usr/local/cuda/bin"
if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")


def _make_gemm_program(M, N, K, block_M, block_N, block_K,
                       dtype="float16", accum_dtype="float"):
    """
    Build a TileLang GEMM program:  C[M, N] = A[M, K] @ B[K, N]

    A = reshaped weight   (Cout, Cin*KH*KW)
    B = im2col matrix     (Cin*KH*KW, H_out*W_out)
    C = output            (Cout, H_out*W_out)
    """
    @T.prim_func
    def gemm_kernel(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


def _compile_kernel(M, N_out, K, dtype_str="float32"):
    """Compile (and cache) a TileLang GEMM kernel for given dimensions.
    
    Parameters
    ----------
    M : int
        Output rows (Cout)
    N_out : int
        Output columns (H_out * W_out)
    K : int
        Inner dimension (Cin * KH * KW)
    dtype_str : str
        Data type string, e.g. "float32" or "float16"
    """
    key = (M, N_out, K, dtype_str)
    if key in _kernel_cache:
        return _kernel_cache[key]

    # Block sizes tuned for sm_75 (T4 / RTX 2080 Ti)
    block_M = 64
    block_N = 64
    block_K = 32

    program = _make_gemm_program(
        M, N_out, K,
        block_M, block_N, block_K,
        dtype=dtype_str,
        accum_dtype="float",
    )

    # --- TileLang v0.1.0 compilation path ---
    mod, params = tilelang.lower(program)
    kernel = tilelang.Profiler(mod, params, [2],
                               tilelang.TensorSupplyType.Integer)

    _kernel_cache[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tilelang_conv2d(x, w, bias=None, stride=1, padding=1):
    """
    Conv2D using im2col + TileLang GEMM.

    Parameters
    ----------
    x : Tensor  (N, Cin, H, W)     – input activations
    w : Tensor  (Cout, Cin, KH, KW) – convolution filters
    bias : Tensor or None
    stride : int
    padding : int

    Returns
    -------
    Tensor  (N, Cout, H_out, W_out)
    """
    # Map torch dtype to tilelang dtype string
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    dtype_str = dtype_map[x.dtype]
    
    orig_dtype = x.dtype
    N_batch, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    # ---- Step 1: im2col via PyTorch unfold (fast CUDA op) ----
    x_col = torch.nn.functional.unfold(
        x, kernel_size=(KH, KW), padding=padding, stride=stride
    )

    # ---- Step 2: reshape weight ----
    w_mat = w.reshape(Cout, -1).contiguous()

    # GEMM dimensions
    M = Cout
    K = Cin * KH * KW
    N_out = H_out * W_out

    # ---- Step 3: compile / fetch cached kernel ----
    kernel = _compile_kernel(M, N_out, K, dtype_str=dtype_str)

    # ---- Step 4: run GEMM for each batch element ----
    outputs = []
    for b in range(N_batch):
        B_mat = x_col[b].contiguous()
        C_mat = kernel(w_mat, B_mat)
        outputs.append(C_mat)

    # ---- Step 5: reshape back to (N, Cout, H_out, W_out) ----
    output = torch.stack(outputs, dim=0).reshape(N_batch, Cout, H_out, W_out)

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)

    return output