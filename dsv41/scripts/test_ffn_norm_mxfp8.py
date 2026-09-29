#!/usr/bin/env python3
"""Parity check for SGLANG_HIP_FFN_NORM_MXFP8: the fused RMSNorm row (rmsnorm_fake_quant_row, shared by
rmsnorm_fake_quant_fp8 and the mHC rmsnorm_with_sinkhorn kernel) emitting MXFP8 next to its bf16 norm vs today's
route = that bf16 norm -> mxfp8_e4m3_quantize (the separate launch before the shared expert's gate_up).
Usage: HIP_VISIBLE_DEVICES=6 PYTHONPATH=<sglang>/python:<aiter> python3 test_ffn_norm_mxfp8.py [--K 5120]
"""
import argparse, torch
from sglang.kernels.ops.quantization.rmsnorm_fake_quant_amd_gfx95 import rmsnorm_fake_quant_fp8
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize

ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=5120)
a = ap.parse_args()
torch.manual_seed(0)
w = (1 + 0.1 * torch.randn(a.K, device="cuda")).to(torch.bfloat16)
ok = True
for M in (1, 6, 12, 48, 96, 300):
    x = (torch.randn(M, a.K, device="cuda") * 4).to(torch.bfloat16)
    act, norm = rmsnorm_fake_quant_fp8(x, w, 1e-6, emit_fp8=True)
    rq, rs = mxfp8_e4m3_quantize(norm)
    q_eq = (act.q.view(torch.uint8) == rq.view(torch.uint8)).float().mean().item()
    s_eq = (act.scale.view(torch.uint8) == rs.view(torch.uint8)).float().mean().item()
    ok &= q_eq == 1.0 and s_eq == 1.0
    print(f"M={M:3d} fp8_codes_equal={q_eq:.6f} scales_equal={s_eq:.6f}")
print("ALL BIT-EXACT" if ok else "MISMATCH")
