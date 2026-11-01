#!/usr/bin/env python3
"""Can aiter's FlyDSL hgemm (gemm_a16w16_gfx950) hold tgemm speed with 8 waves (512 threads = the mono
CTA)? Sweeps 8-wave configs at decode M on the K3.1 KDA-layer shapes, graph-replayed with weights
rotating past the MALL (same bench as decode_gemm_microbench.py). 1 GPU, server down.
Usage: hgemm_8wave_sweep.py [M=128] [N:K,...]"""
import itertools
import os
import sys

import torch

os.environ.setdefault("SGLANG_USE_AITER", "1")
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16  # noqa: E402
from aiter.tuned_gemm import tgemm  # noqa: E402



def bench(fn, ws, iters=48):
    """decode_gemm_microbench.bench (importing it runs its sweep)."""
    for w in ws[:3]:
        fn(w)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for w in ws:
            fn(w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
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

M = int(sys.argv[1]) if len(sys.argv) > 1 else 128
SHAPES = [tuple(map(int, s.split(":"))) for s in (sys.argv[2] if len(sys.argv) > 2 else
          "6288:7168,6016:7168,7168:3584,7168:1536").split(",")]
# (block_m, block_n, m_waves, n_waves, k_waves): 8 waves
WAVES = [(128, 128, 2, 4, 1), (128, 128, 4, 2, 1), (128, 64, 4, 2, 1), (128, 64, 2, 2, 2),
         (128, 96, 4, 2, 1), (64, 128, 2, 4, 1), (128, 128, 2, 2, 2)]
for n, k in SHAPES:
    ncopy = max(4, -(-640 * 2**20 // (n * k * 2)))
    ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
    x = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ ws[0].float().T
    tg = bench(lambda w: tgemm.mm(x, w, None, otype=torch.bfloat16), ws)
    res = []
    for (bm, bn, mw, nw, kw), sk, st in itertools.product(WAVES, (1, 2, 4), (2, 3, 4)):
        kw_ = dict(block_m=bm, block_n=bn, block_k=64, stages=st, split_k=sk, m_waves=mw, n_waves=nw,
                   k_waves=kw)
        fn = lambda w, kw_=kw_: gemm_a16w16(x, w.t(), user_kwargs=kw_, layout="nt")  # noqa: E731
        try:
            err = ((fn(ws[0]).float() - ref).norm() / ref.norm()).item()
            if err > 1e-2:
                continue
            res.append((bench(fn, ws), f"t{bm}x{bn}x64x{st} ks{sk} w{mw}x{nw}x{kw}"))
        except Exception as e:  # noqa: BLE001
            if os.environ.get("SHOW_ERR"):
                print("  skip", kw_, type(e).__name__, str(e)[:150], flush=True)
            continue
    res.sort()
    print(f"M={M} N={n} K={k} tgemm {tg:.1f} us | 8-wave best: "
          + " | ".join(f"{name} {us:.1f}" for us, name in res[:3]) + f" ({len(res)} ok)", flush=True)
    del ws
    torch.cuda.empty_cache()
