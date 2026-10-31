#!/usr/bin/env python3
"""#3 custom skinny-M bf16 GEMM vs the server's tgemm path, graph-replayed, MALL-busting weights.
Usage: AITER_CONFIG_GEMM_BF16=<server csv> skinny_gemm_bench.py [M=128] [N:K,...] [splits=4,8,12] [depths=1,2]"""
import os
import sys

import torch

os.environ.setdefault("SGLANG_USE_AITER", "1")
from aiter.tuned_gemm import tgemm  # noqa: E402

from sglang.kernels.ops.gemm.bf16_skinny_gemm_gfx950 import bf16_skinny_gemm  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 128
SHAPES = [tuple(map(int, s.split(":"))) for s in (sys.argv[2] if len(sys.argv) > 2 else "6016:7168,6288:7168,7168:3584,7168:1536").split(",")]
SPLITS = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "4,8,12").split(",")]
DEPTHS = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else "1,2").split(",")]


def bench(fn, ws, iters=48):
    for w in ws:
        fn(w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(iters):
            fn(ws[i % len(ws)])
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(5):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / (5 * iters) * 1000


for N, K in SHAPES:
    ncopy = max(4, -(-640 * 2**20 // (N * K * 2)))
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ ws[0].float().T
    mb = N * K * 2 / 1e6
    us = bench(lambda w: tgemm.mm(x, w, None, otype=torch.bfloat16), ws)
    print(f"M={M} N={N:6d} K={K:5d} tgemm            {us:6.1f} us {mb / us:5.2f} TB/s", flush=True)
    for d in DEPTHS:
        for s in SPLITS:
            try:
                o = bf16_skinny_gemm(x, ws[0], splits=s, w_depth=d)
                err = ((o.float() - ref).norm() / ref.norm()).item()
                o2 = bf16_skinny_gemm(x, ws[0], splits=s, w_depth=d)
                us = bench(lambda w: bf16_skinny_gemm(x, w, splits=s, w_depth=d), ws)
                print(f"M={M} N={N:6d} K={K:5d} skinny d{d} s{s:<3d}  {us:6.1f} us {mb / us:5.2f} TB/s  err {err:.1e}"
                      f"{'' if torch.equal(o, o2) else '  NONDETERMINISTIC'}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"M={M} N={N} K={K} skinny d{d} s{s} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    del ws
    torch.cuda.empty_cache()
