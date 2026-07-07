# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense (non-grouped) A8W4 GEMM for gfx1250.

``Y = X @ W^T`` with MXFP8 activations (FP8 e4m3 + per-1x32 e8m0 scale) and
MXFP4 weights (FP4 e2m1 packed + per-1x32 e8m0 scale). This is the single-matrix
counterpart of the grouped MoE ``compile_a8w4_gemm`` path in
``kernels/gemm_mxscale_gfx1250.py`` and exists so plain ``nn.Linear`` layers of
an MXFP4 (W4A4) checkpoint can run on hardware that lacks the fp4-activation
WMMA scale instruction (``V_WMMA_SCALE_F32_32X16X128_F4``, e.g. gfx1250): the
activation is quantized to FP8 instead of FP4, so the GEMM uses the FP8 WMMA
scale path.

Data / scale layout consumed by the kernel (mirrors the grouped path):
  * A activation: FP8 e4m3, shape ``(M, K)``.
  * A scale: e8m0 uint8, per-1x32, preshuffled to the warp-tile layout
    (``_preshuffle_a_scale``). Identity when ``tile_m // m_warp == 16``.
  * B weight: FP4 packed uint8, shape ``(N, K // 2)`` -- passed raw, the kernel
    TDM descriptor handles the WMMA tiling (no weight preshuffle needed).
  * B scale: e8m0 uint8, per-1x32, preshuffled to the n32k4 layout
    (``shuffle_scale_n32k4``), shape ``(N // 32, (K // 32) * 32)``.
"""

from __future__ import annotations

import functools

import torch

from aiter.utility import dtypes

OCP_MX_BLOCK_SIZE = 32


@functools.lru_cache(maxsize=1024)
def _compile_dense_a8w4_gemm(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    out_dtype: str,
):
    from aiter.ops.flydsl.kernels.gemm_mxscale_gfx1250 import compile_a8w4_gemm

    return compile_a8w4_gemm(
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        out_dtype=out_dtype,
        # Dense single GEMM: no batching / masking / fused activation.
        grouped_masked_m=False,
        batch_count=1,
    )


def _preshuffle_a_scale(
    scale: torch.Tensor, warp_tile_m: int, tile_k: int
) -> torch.Tensor:
    """Preshuffle the raw per-token e8m0 A-scale ``(M, K//32)`` into the warp
    tile layout the kernel expects. Identity when ``warp_tile_m == 16``."""
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        _grouped_a8w4_preshuffle_e8m0_scale,
    )

    scale_u8 = scale.view(torch.uint8).contiguous()
    M, k_scale = scale_u8.shape
    wmma_rep = int(warp_tile_m) // 16
    if wmma_rep <= 1:
        # No row folding: the warp owns a single 16-row WMMA tile, so the raw
        # (M, K//32) layout is already what the kernel reads.
        return scale_u8
    scale_k_per_tile = int(tile_k) // OCP_MX_BLOCK_SIZE
    shuffled = _grouped_a8w4_preshuffle_e8m0_scale(
        scale_u8.view(1, M, k_scale),
        warp_tile=int(warp_tile_m),
        scale_k_per_tile=scale_k_per_tile,
    )
    return shuffled.view(M // wmma_rep, k_scale * wmma_rep)


def preshuffle_a8w4_weight_scale(w_scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Preshuffle a raw ``(N, K//32)`` e8m0 weight scale into the n32k4 layout
    ``(N//32, (K//32)*32)`` consumed by the gfx1250 A8W4 GEMM. Safe to call once
    at weight-load time."""
    from aiter.ops.shuffle import shuffle_scale_n32k4

    s = w_scale_e8m0.view(torch.uint8).contiguous()
    if s.ndim != 2:
        raise ValueError(f"weight scale must be 2D (N, K//32), got {tuple(s.shape)}")
    N, k_scale = s.shape
    shuffled = shuffle_scale_n32k4(s.view(1, N, k_scale))
    return shuffled.view(N // 32, k_scale * 32)


_OUT_DTYPE_STR = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
}


def run_gemm_a8w4_gfx1250(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale_e8m0: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    w_scale_preshuffled: bool = False,
    tile_m: int = 16,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 1,
    n_warp: int = 2,
    num_buffers: int = 2,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``Y = X @ W^T`` (A8W4) on gfx1250.

    Args:
        x: activation, ``(M, K)`` bf16/fp16.
        w_packed: FP4 weight, ``(N, K//2)`` uint8 (2 fp4 per byte), raw layout.
        w_scale_e8m0: e8m0 weight scale, ``(N, K//32)`` uint8, raw layout unless
            ``w_scale_preshuffled`` is set (then already n32k4).
        out_dtype: output dtype (bf16 or f16).
        w_scale_preshuffled: True if ``w_scale_e8m0`` is already n32k4-shuffled.
        out: optional preallocated ``(M, N)`` output.
    """
    from aiter.ops.triton.quant import dynamic_mxfp8_quant
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    if out_dtype not in _OUT_DTYPE_STR:
        raise ValueError(f"out_dtype must be bf16 or f16, got {out_dtype}")

    orig_shape = x.shape
    if x.dim() != 2:
        x = x.reshape(-1, orig_shape[-1])
    M, K = x.shape
    N, Kp = w_packed.shape
    if Kp != K // 2:
        raise ValueError(
            f"weight K mismatch: x has K={K} (expects K//2={K // 2}), w has {Kp}"
        )
    if K % tile_k != 0:
        raise ValueError(f"K={K} must be divisible by tile_k={tile_k}")
    if N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_n={tile_n}")

    # MXFP8 activation quant: FP8 values + per-1x32 e8m0 scale.
    a_fp8, a_scale = dynamic_mxfp8_quant(x.contiguous(), quant_dtype=dtypes.fp8)
    warp_tile_m = tile_m // m_warp
    a_scale_shuf = _preshuffle_a_scale(a_scale, warp_tile_m, tile_k)

    if w_scale_preshuffled:
        b_scale_shuf = w_scale_e8m0.view(torch.uint8).contiguous()
    else:
        b_scale_shuf = preshuffle_a8w4_weight_scale(w_scale_e8m0)

    if out is None:
        y = torch.empty(M, N, dtype=out_dtype, device=x.device)
    else:
        y = out.reshape(M, N)

    launcher = _compile_dense_a8w4_gemm(
        N=int(N),
        K=int(K),
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        m_warp=int(m_warp),
        n_warp=int(n_warp),
        num_buffers=int(num_buffers),
        out_dtype=_OUT_DTYPE_STR[out_dtype],
    )
    _run_compiled(
        launcher,
        y,
        a_fp8,
        w_packed.view(torch.uint8).contiguous(),
        a_scale_shuf,
        b_scale_shuf,
        int(M),
        int(N),
        torch.cuda.current_stream(),
    )

    if len(orig_shape) != 2:
        return y.view(*orig_shape[:-1], N)
    return y


__all__ = [
    "run_gemm_a8w4_gfx1250",
    "preshuffle_a8w4_weight_scale",
]
