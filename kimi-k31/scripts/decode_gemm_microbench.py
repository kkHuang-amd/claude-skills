#!/usr/bin/env python3
"""#3 decode bf16 GEMMs (per-rank TP8 shapes from AITER_LOG_TUNED_CONFIG), 1 gfx950 GPU, server down.
Times the server's tgemm.mm path (same tuned csv), hipBLASLt (torch.mm) and aiter Triton a16w16.
Weights rotate over enough copies to exceed the 256 MB MALL, so HBM bandwidth is what is measured.
Usage: AITER_CONFIG_GEMM_BF16=<server csv> decode_gemm_microbench.py [M=128] [N:K,...]"""
import os
import sys

import torch

os.environ.setdefault("SGLANG_USE_AITER", "1")
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16  # noqa: E402
from aiter.tuned_gemm import tgemm  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 128
# (N, K): count per VERIFY step at bs=16 from the c16 trace
SHAPES = {(6016, 7168): 92, (6288, 7168): 69, (7168, 3584): 92, (7168, 1536): 93, (7168, 768): 92,
          (18432, 1536): 25, (2112, 7168): 24, (1536, 7168): 24, (4096, 1536): 24, (1536, 128): 69,
          (128, 7168): 24, (32, 7168): 24}
if len(sys.argv) > 2:
    SHAPES = {tuple(map(int, s.split(":"))): 1 for s in sys.argv[2].split(",")}
dev = "cuda"


def bench(fn, ws, iters=48):
    """Graph-replayed like the server's decode, so host dispatch cost is not timed."""
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
    return st.elapsed_time(en) / (5 * iters) * 1000  # us


tot = {}
for (N, K), cnt in SHAPES.items():
    wbytes = N * K * 2
    ncopy = max(4, -(-640 * 2**20 // wbytes))
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ref = x.float() @ ws[0].float().T
    res = {}
    for name, fn in (("tgemm", lambda w: tgemm.mm(x, w, None, otype=torch.bfloat16)),
                     ("hipblaslt", lambda w: torch.mm(x, w.t())),
                     ("triton", lambda w: gemm_a16w16(x, w, None, dtype=torch.bfloat16))):
        try:
            out = fn(ws[0])
            err = ((out.float() - ref).norm() / ref.norm()).item()
            us = bench(fn, ws)
            res[name] = us
            tot[name] = tot.get(name, 0.0) + us * cnt
            print(f"M={M} N={N:6d} K={K:5d} x{cnt:3d} {name:9s} {us:7.1f} us  {(wbytes + (M * K + M * N) * 2) / us / 1e6:5.2f} TB/s  err {err:.1e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"M={M} N={N} K={K} {name} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
    best = min(res, key=res.get) if res else "-"
    tot["best"] = tot.get("best", 0.0) + (res[best] * cnt if res else 0.0)
    del ws
    torch.cuda.empty_cache()
print("per VERIFY step (ms): " + ", ".join(f"{k} {v / 1000:.2f}" for k, v in tot.items()))
