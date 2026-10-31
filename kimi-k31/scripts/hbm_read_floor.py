#!/usr/bin/env python3
"""Streaming-read floor per weight size: one Triton kernel reading N*K bf16 once (graph-replayed,
rotating copies past the MALL), i.e. the best a single GEMM launch over that weight could do.
Usage: hbm_read_floor.py [N:K,...]"""
import sys

import torch
import triton
import triton.language as tl

SHAPES = [(6016, 7168), (6288, 7168), (7168, 3584), (7168, 1536), (7168, 768), (18432, 1536),
          (2112, 7168), (1536, 7168), (4096, 1536), (1536, 128), (128, 7168), (32, 7168)]
if len(sys.argv) > 1:
    SHAPES = [tuple(map(int, s.split(":"))) for s in sys.argv[1].split(",")]


@triton.jit
def _read(x_ptr, out_ptr, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], tl.float32)
    for i in range(ITERS):
        offs = (pid * ITERS + i) * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc))


def floor_us(N, K):
    n = N * K
    ncopy = max(4, -(-640 * 2**20 // (2 * n)))
    ws = [torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_() for _ in range(ncopy)]
    best = None
    for block, iters in ((4096, 4), (8192, 4), (4096, 8), (8192, 8), (2048, 4)):
        grid = triton.cdiv(n, block * iters)
        out = torch.empty(grid, device="cuda")
        f = lambda w: _read[(grid,)](w, out, n, BLOCK=block, ITERS=iters, num_warps=8)  # noqa: E731
        for w in ws:
            f(w)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(48):
                f(ws[i % ncopy])
        g.replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(5):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        us = st.elapsed_time(en) / 240 * 1000
        best = us if best is None else min(best, us)
    return best


for N, K in SHAPES:
    us = floor_us(N, K)
    print(f"N={N:6d} K={K:5d} {N * K * 2 / 2**20:6.1f} MiB  read floor {us:6.1f} us  {N * K * 2 / us / 1e6:5.2f} TB/s", flush=True)
