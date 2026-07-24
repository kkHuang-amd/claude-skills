#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone validation for the gfx950 masked (deep_gemm-style) MoE stage1.

Exercises ``compile_mixed_moe_gemm1(grouped_masked_m=True, ...)`` via
``flydsl_masked_moe_stage1``: expert-major ``[E, max_m, K]`` activations with a
per-expert live-token count ``masked_m[E]``. Compares the kernel output against
a direct grouped stage1 reference that dequantises the SAME quantised tensors
the kernel consumes (so the only delta is fp32-vs-MFMA accumulation).

Usage:
    python op_tests/test_flydsl_masked_moe_stage1_gfx950.py
    python op_tests/test_flydsl_masked_moe_stage1_gfx950.py --a-scale-one   # skip scale-x
"""

from __future__ import annotations

import argparse
import sys

import torch

import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32, e8m0_shuffle

torch.set_default_device("cuda")

LOGITS_DIFF_TOL = 0.01


def _logits_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    x = actual.double()
    y = expected.double()
    denom = (x * x + y * y).sum() + 1e-8
    return float(((x - y) ** 2).sum() / denom)


def _dequant_w_fp4(w_packed: torch.Tensor, w_scale_e8m0: torch.Tensor) -> torch.Tensor:
    """(E, N, K/2) fp4x2 packed + (E, N, K//32) e8m0 -> (E, N, K) f32."""
    E, N, _ = w_packed.shape
    w_f32 = mxfp4_to_f32(w_packed)  # (E, N, K)
    K = w_f32.shape[-1]
    s = e8m0_to_f32(w_scale_e8m0).view(E, N, K // 32, 1)
    return (w_f32.view(E, N, K // 32, 32) * s).view(E, N, K)


def _quant_a_fp8_per1x32(x: torch.Tensor):
    """(M, K) bf16 -> (fp8 M,K), (e8m0 M, K//32). Per-1x32 mxfp8 quant."""
    block = 32
    dtype_max = 448.0
    M, K = x.shape
    flat = x.contiguous().float().view(M, K // block, block)
    max_abs = flat.abs().amax(dim=-1)
    scale_e8m0 = fp4_utils.f32_to_mx_e8m0_scale(
        max_abs, dtype=fp4_utils.MxDtypeInt.FP8_E4M3
    )
    scale_f32 = e8m0_to_f32(scale_e8m0)
    scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
    scale_f32[scale_f32 == 0] = 1.0
    q = (flat / scale_f32.unsqueeze(-1)).clamp(min=-dtype_max, max=dtype_max)
    q = q.to(dtypes.fp8).view(M, K)
    return q, scale_e8m0.view(M, K // block)


def _dequant_a_fp8(
    a_fp8: torch.Tensor, a_scale_e8m0: torch.Tensor
) -> torch.Tensor:
    """(E, max_m, K) fp8 + (E*max_m, K//32) e8m0 -> (E, max_m, K) f32."""
    E, max_m, K = a_fp8.shape
    s = e8m0_to_f32(a_scale_e8m0).view(E, max_m, K // 32, 1)
    return (a_fp8.to(torch.float32).view(E, max_m, K // 32, 32) * s).view(E, max_m, K)


def _ref_stage1_grouped(
    x_deq: torch.Tensor,  # (E, max_m, K) f32
    w1_deq: torch.Tensor,  # (E, 2*inter, K) f32
    masked_m: torch.Tensor,  # (E,)
    inter_dim: int,
) -> torch.Tensor:
    E, max_m, _ = x_deq.shape
    out = torch.zeros((E, max_m, inter_dim), dtype=torch.float32, device=x_deq.device)
    silu = torch.nn.functional.silu
    for e in range(E):
        m = int(masked_m[e].item())
        if m == 0:
            continue
        acc = x_deq[e, :m] @ w1_deq[e].transpose(0, 1)  # (m, 2*inter)
        gate, up = acc.split([inter_dim, inter_dim], dim=-1)
        out[e, :m] = silu(gate) * up
    return out


def _dequant_w2_fp4(w2_packed, w2_scale_e8m0):
    """(E, model_dim, inter/2) fp4x2 + (E, model_dim, inter//32) e8m0 -> (E, model_dim, inter) f32."""
    E, N, _ = w2_packed.shape
    w_f32 = mxfp4_to_f32(w2_packed)  # (E, N, inter)
    inter = w_f32.shape[-1]
    s = e8m0_to_f32(w2_scale_e8m0).view(E, N, inter // 32, 1)
    return (w_f32.view(E, N, inter // 32, 32) * s).view(E, N, inter)


def run_masked_stage1(
    *,
    E: int = 8,
    max_m: int = 64,
    model_dim: int = 512,
    inter_dim: int = 512,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_scale_one: bool = False,
    seed: int = 0,
) -> bool:
    from aiter.ops.flydsl.grouped_moe_gfx950 import flydsl_masked_moe_stage1

    torch.manual_seed(seed)
    dev = "cuda"
    K = model_dim
    scale_blk = 32

    print(
        f"\n[masked stage1] E={E} max_m={max_m} K={model_dim} inter={inter_dim} "
        f"tile=({tile_m},{tile_n},{tile_k}) a_scale_one={a_scale_one}"
    )

    # ---- weights: fp4 (E, 2*inter, K), e8m0 weight scale (E, 2*inter, K//32) ----
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = (torch.randn((E, 2 * inter_dim, K), dtype=torch.bfloat16) / 8)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, 2 * inter_dim, K // 2)  # packed fp4x2 bytes
    w1_scale = w1_scale.view(E, 2 * inter_dim, K // scale_blk)

    w1_deq = _dequant_w_fp4(w1_qt, w1_scale)

    w1_shuf = shuffle_weight(w1_qt, layout=(16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale.reshape(E * 2 * inter_dim, K // scale_blk))

    # ---- masked per-expert token counts ----
    masked_m = torch.randint(1, max_m + 1, (E,), device=dev, dtype=torch.int32)
    masked_m[0] = max_m  # ensure a full expert
    print(f"  masked_m = {masked_m.tolist()}")

    # ---- activations: expert-major (E, max_m, K) bf16, zero-padded past masked_m ----
    x_bf16 = (torch.randn((E, max_m, K), dtype=torch.bfloat16) / 4)
    for e in range(E):
        x_bf16[e, int(masked_m[e].item()):] = 0

    # per-1x32 mxfp8 quant of the grouped activations
    a_fp8, a_scale = _quant_a_fp8_per1x32(x_bf16.view(E * max_m, K))
    a_fp8 = a_fp8.view(E, max_m, K)
    a_scale = a_scale.view(E * max_m, K // scale_blk)

    if a_scale_one:
        # emulate a_scale_one=True: dequant activation with scale=1 -> use fp8 as-is
        x_deq = a_fp8.to(torch.float32)
        a1_scale_shuf = None
    else:
        x_deq = _dequant_a_fp8(a_fp8, a_scale)
        a1_scale_shuf = e8m0_shuffle(a_scale)

    ref = _ref_stage1_grouped(x_deq, w1_deq, masked_m, inter_dim)

    out = flydsl_masked_moe_stage1(
        a_grouped=a_fp8,
        w1=w1_shuf,
        masked_m=masked_m,
        max_m=max_m,
        w1_scale=w1_scale_shuf,
        a1_scale=a1_scale_shuf,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        act="silu",
        gate_mode="separated",
    )
    torch.cuda.synchronize()

    # compare only the valid rows per expert
    mask = torch.zeros((E, max_m), dtype=torch.bool, device=dev)
    for e in range(E):
        mask[e, : int(masked_m[e].item())] = True
    out_v = out.float()[mask]
    ref_v = ref[mask]

    ld = _logits_diff(out_v, ref_v)
    max_delta = (out_v - ref_v).abs().max().item()
    rel = (out_v - ref_v).norm() / ref_v.norm().clamp(min=1e-12)
    print(
        f"  logits_diff={ld:.4e} (gate<{LOGITS_DIFF_TOL}) "
        f"max_delta={max_delta:.4f} rel_l2={float(rel):.4e}"
    )
    print(f"  ref  sample: {ref_v.reshape(-1)[:6]}")
    print(f"  test sample: {out_v.reshape(-1)[:6]}")
    passed = ld < LOGITS_DIFF_TOL
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed


def run_masked_stage2(
    *,
    E: int = 8,
    max_m: int = 64,
    model_dim: int = 512,
    inter_dim: int = 512,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    seed: int = 0,
) -> bool:
    from aiter.ops.flydsl.grouped_moe_gfx950 import flydsl_masked_moe_stage2

    torch.manual_seed(seed)
    dev = "cuda"
    scale_blk = 32
    print(
        f"\n[masked stage2] E={E} max_m={max_m} model={model_dim} inter={inter_dim} "
        f"tile=({tile_m},{tile_n},{tile_k})"
    )

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)

    # ---- W2: fp4 (E, model_dim, inter), e8m0 scale (E, model_dim, inter//32) ----
    w2 = (torch.randn((E, model_dim, inter_dim), dtype=torch.bfloat16) / 8)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w2_scale = w2_scale.view(E, model_dim, inter_dim // scale_blk)
    w2_deq = _dequant_w2_fp4(w2_qt, w2_scale)

    w2_shuf = shuffle_weight(w2_qt, layout=(16, 16))
    w2_scale_shuf = e8m0_shuffle(w2_scale.reshape(E * model_dim, inter_dim // scale_blk))

    masked_m = torch.randint(1, max_m + 1, (E,), device=dev, dtype=torch.int32)
    masked_m[0] = max_m
    print(f"  masked_m = {masked_m.tolist()}")

    # ---- intermediate activations (E, max_m, inter) bf16, zero past masked_m ----
    a2_bf16 = (torch.randn((E, max_m, inter_dim), dtype=torch.bfloat16) / 4)
    for e in range(E):
        a2_bf16[e, int(masked_m[e].item()):] = 0
    a2_fp8, a2_scale = _quant_a_fp8_per1x32(a2_bf16.view(E * max_m, inter_dim))
    a2_fp8 = a2_fp8.view(E, max_m, inter_dim)
    a2_scale = a2_scale.view(E * max_m, inter_dim // scale_blk)

    x_deq = _dequant_a_fp8(a2_fp8, a2_scale)  # (E, max_m, inter) f32
    a2_scale_shuf = e8m0_shuffle(a2_scale)

    # reference: grouped down-proj (no activation, no weight)
    out_ref = torch.zeros((E, max_m, model_dim), dtype=torch.float32, device=dev)
    for e in range(E):
        m = int(masked_m[e].item())
        if m:
            out_ref[e, :m] = x_deq[e, :m] @ w2_deq[e].transpose(0, 1)

    out = flydsl_masked_moe_stage2(
        a_grouped=a2_fp8,
        w2=w2_shuf,
        masked_m=masked_m,
        max_m=max_m,
        model_dim=model_dim,
        w2_scale=w2_scale_shuf,
        a2_scale=a2_scale_shuf,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    torch.cuda.synchronize()

    mask = torch.zeros((E, max_m), dtype=torch.bool, device=dev)
    for e in range(E):
        mask[e, : int(masked_m[e].item())] = True
    out_v = out.float()[mask]
    ref_v = out_ref[mask]
    ld = _logits_diff(out_v, ref_v)
    max_delta = (out_v - ref_v).abs().max().item()
    rel = (out_v - ref_v).norm() / ref_v.norm().clamp(min=1e-12)
    print(
        f"  logits_diff={ld:.4e} (gate<{LOGITS_DIFF_TOL}) "
        f"max_delta={max_delta:.4f} rel_l2={float(rel):.4e}"
    )
    print(f"  ref  sample: {ref_v.reshape(-1)[:6]}")
    print(f"  test sample: {out_v.reshape(-1)[:6]}")
    passed = ld < LOGITS_DIFF_TOL
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed


def run_masked_e2e(
    *,
    E: int = 8,
    max_m: int = 64,
    model_dim: int = 512,
    inter_dim: int = 512,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    seed: int = 0,
) -> bool:
    """End-to-end: kernel stage1 (grouped) -> quant -> kernel stage2 (grouped),
    vs a full grouped reference (silu stage1 -> down-proj stage2)."""
    from aiter.ops.flydsl.grouped_moe_gfx950 import (
        flydsl_masked_moe_stage1,
        flydsl_masked_moe_stage2,
    )

    torch.manual_seed(seed)
    dev = "cuda"
    K = model_dim
    scale_blk = 32
    print(
        f"\n[masked e2e] E={E} max_m={max_m} K={model_dim} inter={inter_dim} "
        f"tile=({tile_m},{tile_n},{tile_k})"
    )

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)

    # weights
    w1 = torch.randn((E, 2 * inter_dim, K), dtype=torch.bfloat16) / 8
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, 2 * inter_dim, K // 2)
    w1_scale = w1_scale.view(E, 2 * inter_dim, K // scale_blk)
    w1_deq = _dequant_w_fp4(w1_qt, w1_scale)
    w1_shuf = shuffle_weight(w1_qt, layout=(16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale.reshape(E * 2 * inter_dim, K // scale_blk))

    w2 = torch.randn((E, model_dim, inter_dim), dtype=torch.bfloat16) / 8
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w2_scale = w2_scale.view(E, model_dim, inter_dim // scale_blk)
    w2_deq = _dequant_w2_fp4(w2_qt, w2_scale)
    w2_shuf = shuffle_weight(w2_qt, layout=(16, 16))
    w2_scale_shuf = e8m0_shuffle(w2_scale.reshape(E * model_dim, inter_dim // scale_blk))

    masked_m = torch.randint(1, max_m + 1, (E,), device=dev, dtype=torch.int32)
    masked_m[0] = max_m
    print(f"  masked_m = {masked_m.tolist()}")

    # stage1 activations
    x_bf16 = torch.randn((E, max_m, K), dtype=torch.bfloat16) / 4
    for e in range(E):
        x_bf16[e, int(masked_m[e].item()):] = 0
    a_fp8, a_scale = _quant_a_fp8_per1x32(x_bf16.view(E * max_m, K))
    a_fp8 = a_fp8.view(E, max_m, K)
    a_scale = a_scale.view(E * max_m, K // scale_blk)
    x_deq = _dequant_a_fp8(a_fp8, a_scale)
    a1_scale_shuf = e8m0_shuffle(a_scale)

    # kernel stage1
    s1 = flydsl_masked_moe_stage1(
        a_grouped=a_fp8, w1=w1_shuf, masked_m=masked_m, max_m=max_m,
        w1_scale=w1_scale_shuf, a1_scale=a1_scale_shuf,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, act="silu",
    )
    torch.cuda.synchronize()

    # quantize stage1 output -> stage2 input
    a2_fp8, a2_scale = _quant_a_fp8_per1x32(s1.view(E * max_m, inter_dim))
    a2_fp8 = a2_fp8.view(E, max_m, inter_dim)
    a2_scale = a2_scale.view(E * max_m, inter_dim // scale_blk)
    a2_scale_shuf = e8m0_shuffle(a2_scale)

    # kernel stage2
    out = flydsl_masked_moe_stage2(
        a_grouped=a2_fp8, w2=w2_shuf, masked_m=masked_m, max_m=max_m,
        model_dim=model_dim, w2_scale=w2_scale_shuf, a2_scale=a2_scale_shuf,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    torch.cuda.synchronize()

    # reference: grouped stage1 (silu) -> quant -> grouped stage2
    ref_s1 = _ref_stage1_grouped(x_deq, w1_deq, masked_m, inter_dim)
    r2q_fp8, r2q_scale = _quant_a_fp8_per1x32(ref_s1.to(torch.bfloat16).view(E * max_m, inter_dim))
    r2_deq = _dequant_a_fp8(r2q_fp8.view(E, max_m, inter_dim), r2q_scale.view(E * max_m, inter_dim // scale_blk))
    ref = torch.zeros((E, max_m, model_dim), dtype=torch.float32, device=dev)
    for e in range(E):
        m = int(masked_m[e].item())
        if m:
            ref[e, :m] = r2_deq[e, :m] @ w2_deq[e].transpose(0, 1)

    mask = torch.zeros((E, max_m), dtype=torch.bool, device=dev)
    for e in range(E):
        mask[e, : int(masked_m[e].item())] = True
    out_v = out.float()[mask]
    ref_v = ref[mask]
    ld = _logits_diff(out_v, ref_v)
    max_delta = (out_v - ref_v).abs().max().item()
    rel = (out_v - ref_v).norm() / ref_v.norm().clamp(min=1e-12)
    print(
        f"  logits_diff={ld:.4e} (gate<{LOGITS_DIFF_TOL}) "
        f"max_delta={max_delta:.4f} rel_l2={float(rel):.4e}"
    )
    print(f"  ref  sample: {ref_v.reshape(-1)[:6]}")
    print(f"  test sample: {out_v.reshape(-1)[:6]}")
    passed = ld < LOGITS_DIFF_TOL
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed


def _fp8_roundtrip(x_bf16: torch.Tensor) -> torch.Tensor:
    """Per-1x32 mxfp8 quant->dequant round-trip (mirrors kernel activation quant)."""
    M, K = x_bf16.shape
    q, s = _quant_a_fp8_per1x32(x_bf16)
    return _dequant_a_fp8(q.view(1, M, K), s.view(M, K // 32)).view(M, K).to(torch.bfloat16)


def _fp4_roundtrip(x_bf16: torch.Tensor) -> torch.Tensor:
    """Per-1x32 mxfp4 quant->dequant round-trip (mirrors a4w4 stage2 a2 quant)."""
    M, K = x_bf16.shape
    q, s = aiter.get_torch_quant(QuantType.per_1x32)(x_bf16, quant_dtype=dtypes.fp4x2)
    q = q.view(M, K // 2)
    s = s.view(M, K // 32)
    sf = e8m0_to_f32(s).view(M, K // 32, 1)
    return (mxfp4_to_f32(q).view(M, K // 32, 32) * sf).view(M, K).to(torch.bfloat16)


def run_masked_full(
    *,
    E: int = 8,
    tokens: int = 64,
    topk: int = 4,
    model_dim: int = 512,
    inter_dim: int = 512,
    max_m: int = 128,
    tile_m: int = 32,
    seed: int = 0,
) -> bool:
    """End-to-end masked grouped MoE (route+scatter+stage1+stage2+combine) vs
    torch_moe reference (a8w4: per-1x32 fp8 activation, mxfp4 weight)."""
    from aiter.fused_moe import fused_topk, torch_moe_stage1, torch_moe_stage2
    from aiter.ops.flydsl.grouped_moe_gfx950 import flydsl_masked_moe_gfx950

    torch.manual_seed(seed)
    dev = "cuda"
    K = model_dim
    scale_blk = 32
    print(
        f"\n[masked full MoE] E={E} tokens={tokens} topk={topk} K={model_dim} "
        f"inter={inter_dim} max_m={max_m} tile_m={tile_m}"
    )

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)

    hidden = (torch.randn((tokens, K), dtype=torch.bfloat16) / 4)
    w1 = torch.randn((E, 2 * inter_dim, K), dtype=torch.bfloat16) / 8
    w2 = torch.randn((E, model_dim, inter_dim), dtype=torch.bfloat16) / 8

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, 2 * inter_dim, K // 2)
    w1_scale = w1_scale.view(E, 2 * inter_dim, K // scale_blk)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w2_scale = w2_scale.view(E, model_dim, inter_dim // scale_blk)

    w1_shuf = shuffle_weight(w1_qt, layout=(16, 16))
    w2_shuf = shuffle_weight(w2_qt, layout=(16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale.reshape(E * 2 * inter_dim, K // scale_blk))
    w2_scale_shuf = e8m0_shuffle(w2_scale.reshape(E * model_dim, inter_dim // scale_blk))

    score = torch.randn((tokens, E), dtype=torch.float32)
    topk_w, topk_id = fused_topk(hidden, score, topk, True)
    topk_id = topk_id.to(torch.int32)

    counts = torch.bincount(topk_id.reshape(-1), minlength=E)
    if int(counts.max().item()) > max_m:
        print(f"  [skip] max expert count {int(counts.max())} > max_m {max_m}")
        return True

    out = flydsl_masked_moe_gfx950(
        hidden_states=hidden,
        w1_shuf=w1_shuf,
        w2_shuf=w2_shuf,
        w1_scale_shuf=w1_scale_shuf,
        w2_scale_shuf=w2_scale_shuf,
        topk_weight=topk_w,
        topk_ids=topk_id,
        E=E,
        model_dim=model_dim,
        inter_dim=inter_dim,
        max_m=max_m,
        activation="silu",
        tile_m=tile_m,
    )
    torch.cuda.synchronize()

    # reference (a8w4): dequant fp8 activation to bf16, torch stage1 -> quant -> stage2
    hidden_deq = _fp8_roundtrip(hidden)
    a2 = torch_moe_stage1(
        hidden_deq,
        w1_qt,
        w2_qt,
        topk_w,
        topk_id,
        dtype=torch.bfloat16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale.view(dtypes.fp8_e8m0),
        swiglu_limit=None,
    )
    a2_deq = _fp8_roundtrip(a2.view(tokens * topk, inter_dim)).view(tokens, topk, inter_dim)
    ref = torch_moe_stage2(
        a2_deq,
        w1_qt,
        w2_qt,
        topk_w,
        topk_id,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale.view(dtypes.fp8_e8m0),
        a2_scale=None,
        doweight=True,
    )

    ld = _logits_diff(out.float(), ref.float())
    max_delta = (out.float() - ref.float()).abs().max().item()
    rel = (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
    print(
        f"  logits_diff={ld:.4e} (gate<{LOGITS_DIFF_TOL}) "
        f"max_delta={max_delta:.4f} rel_l2={float(rel):.4e}"
    )
    print(f"  ref  sample: {ref.reshape(-1)[:6]}")
    print(f"  test sample: {out.reshape(-1)[:6]}")
    passed = ld < LOGITS_DIFF_TOL
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed


def run_fused_moe_hook(
    *,
    E: int = 8,
    tokens: int = 64,
    topk: int = 4,
    model_dim: int = 512,
    inter_dim: int = 512,
    seed: int = 0,
) -> bool:
    """Validate the fused_moe gfx950 masked hook (AITER_GFX950_MASKED_MOE=1)
    end-to-end vs torch_moe reference."""
    import os

    from aiter.fused_moe import fused_topk, fused_moe, torch_moe_stage1, torch_moe_stage2

    torch.manual_seed(seed)
    K = model_dim
    scale_blk = 32
    print(
        f"\n[fused_moe gfx950 hook] E={E} tokens={tokens} topk={topk} "
        f"K={model_dim} inter={inter_dim}"
    )

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    hidden = (torch.randn((tokens, K), dtype=torch.bfloat16) / 4)
    w1 = torch.randn((E, 2 * inter_dim, K), dtype=torch.bfloat16) / 8
    w2 = torch.randn((E, model_dim, inter_dim), dtype=torch.bfloat16) / 8

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, 2 * inter_dim, K // 2)
    w1_scale = w1_scale.view(E, 2 * inter_dim, K // scale_blk)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w2_scale = w2_scale.view(E, model_dim, inter_dim // scale_blk)

    score = torch.randn((tokens, E), dtype=torch.float32)
    topk_w, topk_id = fused_topk(hidden, score, topk, True)
    topk_id = topk_id.to(torch.int32)

    prev = os.environ.get("AITER_GFX950_MASKED_MOE")
    os.environ["AITER_GFX950_MASKED_MOE"] = "1"
    try:
        out = fused_moe(
            hidden,
            w1_qt.view(dtypes.fp4x2),
            w2_qt.view(dtypes.fp4x2),
            topk_w,
            topk_id,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            activation=ActivationType.Silu,
            gate_mode="separated",
            dtype=dtypes.bf16,
        )
    finally:
        if prev is None:
            os.environ.pop("AITER_GFX950_MASKED_MOE", None)
        else:
            os.environ["AITER_GFX950_MASKED_MOE"] = prev
    torch.cuda.synchronize()

    hidden_deq = _fp8_roundtrip(hidden)
    a2 = torch_moe_stage1(
        hidden_deq, w1_qt, w2_qt, topk_w, topk_id,
        dtype=torch.bfloat16, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, a1_scale=None,
        w1_scale=w1_scale.view(dtypes.fp8_e8m0), swiglu_limit=None,
    )
    a2_deq = _fp8_roundtrip(a2.view(tokens * topk, inter_dim)).view(tokens, topk, inter_dim)
    ref = torch_moe_stage2(
        a2_deq, w1_qt, w2_qt, topk_w, topk_id,
        dtype=torch.bfloat16, quant_type=QuantType.per_1x32,
        w2_scale=w2_scale.view(dtypes.fp8_e8m0), a2_scale=None, doweight=True,
    )

    ld = _logits_diff(out.float(), ref.float())
    rel = (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
    print(f"  logits_diff={ld:.4e} (gate<{LOGITS_DIFF_TOL}) rel_l2={float(rel):.4e}")
    passed = ld < LOGITS_DIFF_TOL
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed


def run_masked_recv(
    *,
    E: int = 8,  # local experts
    max_m: int = 128,
    model_dim: int = 512,
    inter_dim: int = 512,
    tile_m: int = 32,
    topk: int = 6,
    recv_mode: str = "fp4",  # "fp4" | "fp8" | "bf16"
    seed: int = 0,
) -> bool:
    """Validate the mori flat-recv bridge primitive
    (`flydsl_masked_moe_gfx950_recv`): token-major recv with GLOBAL topk ids ->
    global->local route -> masked s1/s2 -> WEIGHTED gather-reduce combine, vs a
    per-token weighted reference (matches the sorted fused_moe(mori) semantics)."""
    from aiter.ops.flydsl.grouped_moe_gfx950 import flydsl_masked_moe_gfx950_recv

    torch.manual_seed(seed)
    dev = "cuda"
    K = model_dim
    scale_blk = 32
    expert_base = E  # local block = global [E, 2E); ids < E are non-local (dropped)
    E_global = 2 * E
    M = 96

    print(
        f"\n[masked recv] E_local={E} E_global={E_global} base={expert_base} "
        f"M={M} topk={topk} max_m={max_m} "
        f"K={model_dim} inter={inter_dim} recv={recv_mode}"
    )

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = torch.randn((E, 2 * inter_dim, K), dtype=torch.bfloat16) / 8
    w2 = torch.randn((E, model_dim, inter_dim), dtype=torch.bfloat16) / 8
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, 2 * inter_dim, K // 2)
    w1_scale = w1_scale.view(E, 2 * inter_dim, K // scale_blk)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w2_scale = w2_scale.view(E, model_dim, inter_dim // scale_blk)
    w1_deq = _dequant_w_fp4(w1_qt, w1_scale)
    w2_deq = _dequant_w2_fp4(w2_qt, w2_scale)
    w1_shuf = shuffle_weight(w1_qt, layout=(16, 16))
    w2_shuf = shuffle_weight(w2_qt, layout=(16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale.reshape(E * 2 * inter_dim, K // scale_blk))
    w2_scale_shuf = e8m0_shuffle(w2_scale.reshape(E * model_dim, inter_dim // scale_blk))

    # token-major recv: each row a token with `topk` GLOBAL ids + weights
    recv_topk_ids = torch.randint(0, E_global, (M, topk), device=dev, dtype=torch.int32)
    recv_topk_w = torch.rand((M, topk), device=dev, dtype=torch.float32) + 0.1
    # padding stress: last 24 rows are zero-inited padding (topk_ids=base -> local
    # expert 0, weight 0). Must be EXCLUDED (weight>0) so they neither corrupt
    # expert 0 nor consume its max_m slots.
    # PADDING = rows >= total_recv (prefix rule). Give them local ids + STALE
    # NONZERO weights to verify the prefix (NOT weight) excludes them.
    total_recv = M - 24
    recv_topk_ids[total_recv:] = expert_base  # local expert 0
    recv_topk_w[total_recv:] = 0.5  # stale nonzero (would fool a weight>0 filter)
    recv_bf16 = (torch.randn((M, K), dtype=torch.bfloat16) / 4)
    x_deq_ref = None
    if recv_mode == "fp4":
        # a4w4: recv is mxfp4 (fp4x2 + per-1x32 e8m0), like mori mxfp4 dispatch
        recv_fp4, recv_scale_f4 = torch_quant(recv_bf16, quant_dtype=dtypes.fp4x2)
        recv_fp4 = recv_fp4.view(M, K // 2)
        recv_scale_f4 = recv_scale_f4.view(M, K // scale_blk)
        recv_in, scale_in = recv_fp4, recv_scale_f4
        s = e8m0_to_f32(recv_scale_f4).view(M, K // scale_blk, 1)
        x_deq_ref = (mxfp4_to_f32(recv_fp4).view(M, K // scale_blk, 32) * s).view(M, K)
    else:
        recv_fp8, recv_scale_t = _quant_a_fp8_per1x32(recv_bf16)
        if recv_mode == "bf16":
            recv_in, scale_in = recv_bf16, None
        else:  # fp8
            recv_in, scale_in = recv_fp8, recv_scale_t
        x_deq_ref = _dequant_a_fp8(
            recv_fp8.view(1, M, K), recv_scale_t.view(M, K // scale_blk)
        ).view(M, K)

    out = flydsl_masked_moe_gfx950_recv(
        recv_hidden=recv_in,
        recv_scale=scale_in,
        recv_topk_ids=recv_topk_ids,
        recv_topk_weights=recv_topk_w,
        w1_shuf=w1_shuf,
        w2_shuf=w2_shuf,
        w1_scale_shuf=w1_scale_shuf,
        w2_scale_shuf=w2_scale_shuf,
        E=E,
        expert_base=expert_base,
        total_recv=total_recv,
        model_dim=model_dim,
        inter_dim=inter_dim,
        max_m=max_m,
        activation="silu",
        tile_m=tile_m,
    )
    torch.cuda.synchronize()

    # reference: per-token weighted sum over LOCAL experts; padding rows (>=
    # total_recv) are ZERO (matches default's num_valid clipping).
    x_deq = x_deq_ref  # dequant of the recv activation (fp4 or fp8 round-trip)
    silu = torch.nn.functional.silu
    ref = torch.zeros((M, model_dim), dtype=torch.float32, device=dev)
    for m in range(total_recv):
        acc = torch.zeros(model_dim, dtype=torch.float32, device=dev)
        for k in range(topk):
            loc = int(recv_topk_ids[m, k].item()) - expert_base
            if not (0 <= loc < E):
                continue
            g_u = x_deq[m] @ w1_deq[loc].transpose(0, 1)
            gate, up = g_u.split([inter_dim, inter_dim], dim=-1)
            s1 = (silu(gate) * up).to(torch.bfloat16)
            # a2 dtype must match the masked primitive (default fp4 a4w4-stage2)
            import os as _os
            if _os.environ.get("SGLANG_MORI_MASKED_A2_DTYPE", "fp4").lower() == "fp8":
                s1d = _fp8_roundtrip(s1.view(1, inter_dim)).view(inter_dim)
            else:
                s1d = _fp4_roundtrip(s1.view(1, inter_dim)).view(inter_dim)
            s2 = s1d.float() @ w2_deq[loc].transpose(0, 1)
            acc += recv_topk_w[m, k] * s2
        ref[m] = acc

    ld = _logits_diff(out.float(), ref)
    rel = (out.float() - ref).norm() / ref.norm().clamp(min=1e-12)
    print(f"  logits_diff={ld:.4e} (gate<{LOGITS_DIFF_TOL}) rel_l2={float(rel):.4e}")
    passed = ld < LOGITS_DIFF_TOL
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-E", "--experts", type=int, default=8)
    parser.add_argument("--max-m", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--inter-dim", type=int, default=512)
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument(
        "--a-scale-one",
        action="store_true",
        help="skip scale-x (a_scale_one=True) to validate GEMM/scheduler first",
    )
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "e2e", "full", "hook", "recv", "both", "all"),
        default="all",
        help="which masked stage(s) to validate",
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--bf16-recv", action="store_true", help="recv path: bf16 dispatch")
    parser.add_argument("--recv-mode", choices=("fp4", "fp8", "bf16"), default="fp4")
    args = parser.parse_args()

    ok = True
    if args.stage in ("stage1", "both", "all"):
        ok &= run_masked_stage1(
            E=args.experts,
            max_m=args.max_m,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            tile_m=args.tile_m,
            a_scale_one=args.a_scale_one,
        )
    if args.stage in ("stage2", "both", "all"):
        ok &= run_masked_stage2(
            E=args.experts,
            max_m=args.max_m,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            tile_m=args.tile_m,
        )
    if args.stage in ("e2e", "all"):
        ok &= run_masked_e2e(
            E=args.experts,
            max_m=args.max_m,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            tile_m=args.tile_m,
        )
    if args.stage in ("full", "all"):
        ok &= run_masked_full(
            E=args.experts,
            tokens=args.tokens,
            topk=args.topk,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            max_m=args.max_m,
            tile_m=args.tile_m,
        )
    if args.stage in ("recv", "all"):
        ok &= run_masked_recv(
            E=args.experts,
            max_m=args.max_m,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            tile_m=args.tile_m,
            recv_mode=("bf16" if args.bf16_recv else args.recv_mode),
        )
    if args.stage in ("hook", "all"):
        ok &= run_fused_moe_hook(
            E=args.experts,
            tokens=args.tokens,
            topk=args.topk,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
