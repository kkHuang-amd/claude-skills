"""Gluon a16w16 (gfx1250) decode-shape bench: non-persistent config vs persistent + NUM_KSPLIT.

Cold weights (rotating copies > --cold-mb), CUDA graph of len(copies) launches, replayed; prints us/launch.
Usage: HIP_VISIBLE_DEVICES=0 python3 a16w16_splitk_bench.py --M 2 --N 2304 --K 16384
"""

import argparse
import json

import torch
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=2)
ap.add_argument("--N", type=int, default=2304)
ap.add_argument("--K", type=int, default=16384)
ap.add_argument("--cold-mb", type=int, default=2048)
ap.add_argument("--replays", type=int, default=20)
ap.add_argument("--configs", default="", help="JSON list of [label, persistent, config]")
a = ap.parse_args()

M, N, K = a.M, a.N, a.K
wbytes = N * K * 2
ncopy = max(2, min(24, a.cold_mb * 2**20 // wbytes))
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
ys = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(ncopy)]
ref = x.float() @ ws[0].float().T

base = {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 512, "NUM_BUFFERS": 5, "num_warps": 2, "waves_per_eu": 1}
cands = [["default", False, None], ["nonpers_best", False, dict(base, kernel_type="compute_bound")]]
for ks in (1, 2, 4, 8, 16):
    for bn in (16, 32):
        cands.append([f"pers_bn{bn}_ks{ks}", True, dict(base, BLOCK_N=bn, NUM_KSPLIT=ks, kernel_type="bandwidth_bound")])
if a.configs:
    cands = json.loads(a.configs)


def run(i, pers, cfg):
    kw = {"y": ys[i], "dtype": torch.bfloat16}
    if cfg is not None:
        kw["config"] = dict(cfg)
    if pers:
        kw["persistent"] = True
    return gemm_a16w16(x, ws[i], **kw)


print(f"M={M} N={N} K={K} weights {wbytes / 1e6:.1f} MB x {ncopy} copies, HBM floor {wbytes / 8e6:.1f} us @ 8 TB/s")
for label, pers, cfg in cands:
    try:
        out = run(0, pers, cfg)
        err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
        for i in range(ncopy):
            run(i, pers, cfg)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(ncopy):
                run(i, pers, cfg)
        g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(a.replays):
            s.record()
            g.replay()
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e) * 1000 / ncopy)
        ts.sort()
        us = ts[len(ts) // 2]
        print(f"{label:22s} {us:7.2f} us  {wbytes / us / 1e6:5.2f} TB/s  relerr {err:.1e}")
    except Exception as ex:  # noqa: BLE001
        print(f"{label:22s} ERROR {str(ex)[:150]}")
