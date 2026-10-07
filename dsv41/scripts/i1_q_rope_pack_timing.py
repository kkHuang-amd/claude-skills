#!/usr/bin/env python3
"""I1 timing at prefill-like positions (contiguous, as a chunk of one request) + bitwise stress over seeds.
  PYTHONPATH=/sgl-workspace/sglang-i1/python HIP_VISIBLE_DEVICES=0 python3 i1_q_rope_pack_timing.py
Env: SEEDS (default 20) bitwise stress cases at T=16384 H=32 with NW_STRESS (default 2) warps.
"""
import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("chk", os.path.join(HERE, "i1_q_rope_pack_check.py"))
from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import index_q_rope_pack_flydsl, pack_fp4_query_flydsl
from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4

dev, D, RD, MAXPOS = "cuda", 128, 64, 1 << 20


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


torch.manual_seed(1)
freqs = torch.polar(torch.ones(MAXPOS, RD // 2), torch.rand(MAXPOS, RD // 2) * 6.283).to(dev)
H = 32
for T, start in ((4096, 0), (4096, 126976), (16384, 0), (16384, 114688)):
    q = (torch.randn(T, H * D, device=dev) * 3).to(torch.bfloat16)
    pos = torch.arange(start, start + T, device=dev)
    old = lambda: pack_fp4_query_flydsl(_rope_fq4(q.view(T, H, D), freqs, RD, positions=pos))
    t_old = gpu_us(old)
    res = [f"nw{nw} {gpu_us(lambda: index_q_rope_pack_flydsl(q, freqs, pos, RD, num_heads=H, num_warps=nw)):7.1f}"
           for nw in (1, 2, 4)]
    print(f"T={T:>6} pos {start}..: old {t_old:7.1f} us | new " + " | ".join(res) + " us", flush=True)

nw, bad = int(os.environ.get("NW_STRESS", "2")), 0
for seed in range(int(os.environ.get("SEEDS", "20"))):
    torch.manual_seed(100 + seed)
    T = 16384
    q = (torch.randn(T, H * D, device=dev) * (1 + seed % 5)).to(torch.bfloat16)
    pos = torch.randint(0, MAXPOS, (T,), device=dev)
    a = pack_fp4_query_flydsl(_rope_fq4(q.view(T, H, D), freqs, RD, positions=pos))
    b = index_q_rope_pack_flydsl(q, freqs, pos, RD, num_heads=H, num_warps=nw)
    bad += int((a[0] != b[0]).sum()) + int((a[1] != b[1]).sum())
print(f"stress nw={nw}: {os.environ.get('SEEDS', '20')} x T=16384 H=32 -> mismatched bytes {bad}")
