#!/usr/bin/env python3
"""Parity check for the wo_a batched GEMM emitting MXFP8 (emit_fp8) vs today's route = the same GEMM's bf16 output
-> mxfp8_e4m3_quantize (the separate _mxfp8_quant launch before wo_b). Covers split-K and single-chain regimes.
Usage: HIP_VISIBLE_DEVICES=6 PYTHONPATH=<sglang>/python:<aiter> python3 test_wo_a_mxfp8.py [--G 4 --R 1024 --D 4096]
"""
import argparse, torch
from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import batched_gemm_bf16_fp8_grid
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize

ap = argparse.ArgumentParser()
ap.add_argument("--G", type=int, default=4); ap.add_argument("--R", type=int, default=1024); ap.add_argument("--D", type=int, default=4096)
a = ap.parse_args()
torch.manual_seed(0)
w = (torch.randn(a.G, a.R, a.D, device="cuda") * 0.02).to(torch.bfloat16)
ok = True
for T in (1, 6, 12, 48, 64, 96, 200):
    x = torch.randn(T, a.G, a.D, device="cuda").to(torch.bfloat16)
    for split_k in ((None, True, False) if T <= 64 else (None, False)):
        ref = batched_gemm_bf16_fp8_grid(x, w, fp8_grid=False, split_k=split_k)
        rq, rs = mxfp8_e4m3_quantize(ref)
        nq, ns = batched_gemm_bf16_fp8_grid(x, w, fp8_grid=False, split_k=split_k, emit_fp8=True)
        q_eq = (nq.view(torch.uint8) == rq.view(torch.uint8)).float().mean().item()
        s_eq = (ns == rs.view(torch.uint8)).float().mean().item()
        ok &= q_eq == 1.0 and s_eq == 1.0
        print(f"T={T:3d} split_k={str(split_k):5s} fp8_codes_equal={q_eq:.6f} scales_equal={s_eq:.6f}")
print("ALL BIT-EXACT" if ok else "MISMATCH")
