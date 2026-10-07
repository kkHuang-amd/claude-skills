#!/usr/bin/env python3
"""I1 (INDEXER_COST_1007.md): fused prefill index-Q kernel `index_q_rope_pack_flydsl` vs the current
`pack_fp4_query_flydsl(_rope_fq4(...))`. Bitwise check (q_fp4 + q_scale) and GPU time (CUDA-graph replay).
  PYTHONPATH=/sgl-workspace/sglang-i1/python HIP_VISIBLE_DEVICES=0 python3 i1_q_rope_pack_check.py
Env: SWEEP=1 also times heads_per_program x num_warps variants at T=16384; NW=<num_warps> for the check (default 1).
"""
import os

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
    index_q_rope_pack_flydsl,
    pack_fp4_query_flydsl,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4

dev, D, RD, MAXPOS = "cuda", 128, 64, 1 << 20
NW = int(os.environ.get("NW", "1"))


def gpu_us(fn, iters=20):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(3):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (3 * iters)


torch.manual_seed(0)
freqs = torch.polar(torch.ones(MAXPOS, RD // 2), torch.rand(MAXPOS, RD // 2) * 6.283).to(dev)
assert freqs.dtype == torch.complex64
ok_all = True
for H in (32, 64, 16):
    for T in (1, 7, 128, 4096, 16384):
        q = (torch.randn(T, H * D, device=dev) * 3).to(torch.bfloat16)
        pos = torch.randint(0, MAXPOS, (T,), device=dev)
        if T > 4:
            pos[3] = MAXPOS + 5  # out-of-range rows read entry 0 on both paths
            pos[4] = -1

        def old():
            return pack_fp4_query_flydsl(_rope_fq4(q.view(T, H, D), freqs, RD, positions=pos))

        def new():
            return index_q_rope_pack_flydsl(q, freqs, pos, RD, num_heads=H, num_warps=NW)

        (a_p, a_s), (b_p, b_s) = old(), new()
        ok = torch.equal(a_p, b_p) and torch.equal(a_s, b_s)
        ok_all &= ok
        line = f"H={H:>2} T={T:>6}: bitwise {'OK ' if ok else 'MISMATCH'}"
        if not ok:
            line += f" payload diff {int((a_p != b_p).sum())} scale diff {int((a_s != b_s).sum())}"
        if T >= 4096:
            line += f" | old {gpu_us(old):8.1f} us  new {gpu_us(new):8.1f} us"
        print(line, flush=True)

if os.environ.get("SWEEP") == "1":
    T, H = 16384, 32
    q = torch.randn(T, H * D, device=dev).to(torch.bfloat16)
    pos = torch.randint(0, MAXPOS, (T,), device=dev)
    for hpp in (1, 2, 4, 8, 16, 32):
        for nw in (1, 2, 4):
            us = gpu_us(lambda: index_q_rope_pack_flydsl(
                q, freqs, pos, RD, num_heads=H, heads_per_program=hpp, num_warps=nw))
            print(f"sweep T={T} H={H} heads_per_program={hpp:>2} num_warps={nw}: {us:8.1f} us", flush=True)
print("PASS" if ok_all else "FAIL")
