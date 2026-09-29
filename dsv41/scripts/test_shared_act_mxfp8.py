#!/usr/bin/env python3
"""Parity check for SGLANG_HIP_SHARED_ACT_MXFP8: Triton silu_and_mul_clamp emitting MXFP8 directly (multi-block,
inter > 1024) vs today's route = aiter fused_clamp_act_mul (bf16) -> mxfp8_e4m3_quantize.
Reports exact-match fraction of fp8 codes / ue8m0 scales and max relative error after dequant.
Usage: HIP_VISIBLE_DEVICES=6 PYTHONPATH=<sglang>/python:<aiter> python3 test_shared_act_mxfp8.py [--inter 1152]
"""
import argparse, torch
from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
from sglang.kernels.ops.activation.silu_and_mul_clamp_hip import silu_and_mul_clamp_triton
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize, dequant_mxfp8_to_bf16

ap = argparse.ArgumentParser(); ap.add_argument("--inter", type=int, default=1152); ap.add_argument("--limit", type=float, default=10.0)
a = ap.parse_args()
torch.manual_seed(0)
for M in (1, 6, 12, 48, 96):
    gu = (torch.randn(M, 2 * a.inter, device="cuda") * 3).to(torch.bfloat16)
    ref_bf16 = fused_clamp_act_mul(gu, swiglu_limit=a.limit, activation="silu")
    rq, rs = mxfp8_e4m3_quantize(ref_bf16)
    new = silu_and_mul_clamp_triton(gu, a.limit, emit_fp8=True)
    nq, ns = new.q, new.scale
    same_bf16 = (silu_and_mul_clamp_triton(gu, a.limit) == ref_bf16).float().mean().item()
    q_eq = (nq.view(torch.uint8) == rq.view(torch.uint8)).float().mean().item()
    s_eq = (ns.view(torch.uint8) == rs.view(torch.uint8)).float().mean().item()
    dn, dr = dequant_mxfp8_to_bf16(nq, ns).float(), dequant_mxfp8_to_bf16(rq, rs).float()
    rel = ((dn - dr).abs().max() / dr.abs().max().clamp_min(1e-6)).item()
    print(f"M={M:3d} bf16_equal={same_bf16:.4f} fp8_codes_equal={q_eq:.4f} scales_equal={s_eq:.4f} max_rel_err={rel:.2e}")
