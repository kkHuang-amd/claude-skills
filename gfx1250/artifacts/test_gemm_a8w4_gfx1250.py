#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for the dense A8W4 GEMM on gfx1250.

Validates ``run_gemm_a8w4_gfx1250`` (MXFP8 activation x MXFP4 weight) against a
torch emulation that dequantizes the *same* quantized operands the kernel
consumes, so the only remaining error is fp32-vs-WMMA accumulation noise.
"""

from __future__ import annotations

import pytest
import torch

from aiter.utility import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant, dynamic_mxfp8_quant
from aiter.ops.flydsl.gemm_a8w4_gfx1250 import run_gemm_a8w4_gfx1250

SCALE_BLOCK = 32

# fp4 e2m1 lookup table (index = 4-bit code).
MXFP4_TABLE = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _is_gfx1250() -> bool:
    try:
        return "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


def _e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    xf = 2.0 ** (x.to(torch.float32) - 127.0)
    xf[x == 255] = float("nan")
    return xf


def _dequant_mxfp8(x_fp8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    M, K = x_fp8.shape
    v = x_fp8.to(torch.float32).view(M, K // SCALE_BLOCK, SCALE_BLOCK)
    s = _e8m0_to_f32(scale_e8m0).view(M, K // SCALE_BLOCK, 1)
    return (v * s).view(M, K)


def _dequant_mxfp4(w_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    N, Kp = w_packed.shape
    K = Kp * 2
    lut = torch.tensor(MXFP4_TABLE, device=w_packed.device, dtype=torch.float32)
    lo = (w_packed & 0xF).long()
    hi = (w_packed >> 4).long()
    vals = torch.empty(N, K, device=w_packed.device, dtype=torch.float32)
    vals[:, 0::2] = lut[lo]
    vals[:, 1::2] = lut[hi]
    s = _e8m0_to_f32(scale_e8m0).view(N, K // SCALE_BLOCK, 1)
    return (vals.view(N, K // SCALE_BLOCK, SCALE_BLOCK) * s).view(N, K)


def _run_case(M: int, N: int, K: int, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.2

    w_packed, w_scale = dynamic_mxfp4_quant(w)  # (N, K//2) uint8, (N, K//32) uint8

    y = run_gemm_a8w4_gfx1250(x, w_packed, w_scale, out_dtype=torch.bfloat16)

    # Emulate with the SAME activation quantization the kernel performs.
    a_fp8, a_scale = dynamic_mxfp8_quant(x.contiguous(), quant_dtype=dtypes.fp8)
    a_ref = _dequant_mxfp8(a_fp8, a_scale)
    w_ref = _dequant_mxfp4(w_packed, w_scale)
    ref = (a_ref @ w_ref.T).to(torch.float32)

    y_f = y.to(torch.float32)
    rel_l2 = (y_f - ref).norm() / (ref.norm() + 1e-6)
    logits_diff = ((y_f - ref) ** 2).sum() / ((y_f**2).sum() + (ref**2).sum() + 1e-6)
    print(
        f"[a8w4 M={M} N={N} K={K}] rel_l2={rel_l2.item():.4f} "
        f"logits_diff={logits_diff.item():.6f}"
    )
    return rel_l2.item(), logits_diff.item()


@pytest.mark.skipif(not _is_gfx1250(), reason="requires gfx1250")
@pytest.mark.parametrize(
    "M,N,K",
    [
        (16, 128, 256),
        (32, 256, 512),
        (64, 512, 1024),
        (128, 256, 7168),   # DeepSeek dense-MLP-ish K
        (17, 128, 512),     # non-tile-multiple M
    ],
)
def test_gemm_a8w4_gfx1250(M, N, K):
    _, logits_diff = _run_case(M, N, K)
    assert logits_diff < 0.01, f"logits_diff too high: {logits_diff}"


if __name__ == "__main__":
    for (M, N, K) in [(16, 128, 256), (32, 256, 512), (128, 256, 7168), (17, 128, 512)]:
        _run_case(M, N, K)
    print("done")
