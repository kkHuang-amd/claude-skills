#!/usr/bin/env python3
"""12-G4: stage1 mxfp8 quant + scale sort for Kimi-K3.1 (E=896, top-16, latent 3584), 1 GPU.
aiter picks the fused kernel (re-quantizes every token once per route) for M <= 8*256/topk = 128
tokens and the split path (per-token quant + scale sort) above. Times both (CUDA graph) and
checks they produce the same fp8 rows and the same sorted scales.
Usage: moe_quant_sort_microbench.py [M list, default 16,32,64,96,128,192,256,384]"""
import sys

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort_hip, mxfp4_moe_sort_hip, per_1x32_mx_quant_hip

E, TOPK, DIM, BLOCK = 896, 16, 3584, 32
Ms = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "16,32,64,96,128,192,256,384").split(",")]
torch.manual_seed(0)
bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.1


def timed(fn, iters=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000


for M in Ms:
    gating = torch.randn(M, E, device="cuda", dtype=torch.float32)
    tw = torch.empty(M, TOPK, device="cuda", dtype=torch.float32)
    ti = torch.empty(M, TOPK, device="cuda", dtype=torch.int32)
    aiter.biased_grouped_topk(gating, bias, tw, ti, 1, 1, True, 1.0)
    sorted_ids, sorted_w, _, num_valid, _ = moe_sorting(ti, tw, E, DIM, torch.bfloat16, block_size=BLOCK)
    x = torch.randn(M, DIM, device="cuda", dtype=torch.bfloat16)
    scaleN_pad = (DIM // 32 + 7) // 8 * 8
    srows = (sorted_ids.shape[0] + 31) // 32 * 32
    out_f = torch.empty(M, DIM, dtype=dtypes.fp8, device="cuda")
    sc_f = torch.zeros(srows, scaleN_pad, dtype=dtypes.fp8_e8m0, device="cuda")
    sc_s = torch.zeros(srows, scaleN_pad, dtype=dtypes.fp8_e8m0, device="cuda")

    def fused():
        fused_dynamic_mx_quant_moe_sort_hip(out_f, sc_f, x, sorted_ids, num_valid, M, BLOCK, M * TOPK, 32, sorted_w)

    split_out = {}

    def split():
        o, s_tok = per_1x32_mx_quant_hip(x, scale=None, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False,
                                        num_rows=None, num_rows_factor=1)
        mxfp4_moe_sort_hip(sc_s, s_tok, sorted_ids, num_valid, M, DIM)
        split_out["o"] = o

    fused(); split(); torch.cuda.synchronize()
    nv = int(num_valid.flatten()[0])
    same_rows = torch.equal(out_f.view(torch.uint8), split_out["o"].view(torch.uint8))
    same_sc = torch.equal(sc_f.view(torch.uint8)[:nv, : DIM // 32], sc_s.view(torch.uint8)[:nv, : DIM // 32])
    tf, ts = timed(fused), timed(split)
    print(f"M={M:4d} pairs={M * TOPK:5d}  fused {tf:6.2f} us  split {ts:6.2f} us  "
          f"rows equal {same_rows} scales equal {same_sc}", flush=True)
