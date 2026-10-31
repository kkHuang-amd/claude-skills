#!/usr/bin/env python3
"""Kimi-K3.1 MoE prologue on one gfx950 GPU (server must be down): router top-k
(aiter biased_grouped_topk) + aiter moe_sorting, per kernel, at verify token counts.
Shapes per TP8 rank: E=896, top-16, fp32 gating, latent hidden 3584, sort block 32.
Usage: moe_prologue_microbench.py [block_size=32]"""
import sys
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

import aiter
from aiter.fused_moe import moe_sorting

# sglang's runtime patch of aiter's sort; with SGLANG_ROCM_MOE_SORT_ONE_LAUNCH_LARGE unset it
# falls through to aiter's two-phase opus sort past 256 pairs
from sglang.kernels.ops.moe.moe_sorting_small import apply_aiter_small_moe_sort_patch

apply_aiter_small_moe_sort_patch()

E, TOPK, DIM = 896, 16, 3584
BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 32
ITERS = 50
torch.manual_seed(0)
bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.1

for M in tuple(int(x) for x in __import__('os').environ.get('MB_M', '32,128,384').split(',')):
    gating = torch.randn(M, E, device="cuda", dtype=torch.float32)
    tw = torch.empty(M, TOPK, device="cuda", dtype=torch.float32)
    ti = torch.empty(M, TOPK, device="cuda", dtype=torch.int32)

    def step():
        aiter.biased_grouped_topk(gating, bias, tw, ti, 1, 1, True, 1.0)
        return moe_sorting(ti, tw, E, DIM, torch.bfloat16, block_size=BLOCK)

    for _ in range(5):
        out = step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            step()
        torch.cuda.synchronize()
    agg = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type.name == "CUDA" and e.device_time > 0:
            a = agg[e.name[:90]]; a[0] += 1; a[1] += e.device_time
    total = sum(v[1] for v in agg.values()) / ITERS
    print(f"M={M:<4} pairs={M * TOPK:<5} total {total:6.2f} us/iter")
    for n, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
        print(f"    {t / ITERS:6.2f} us x{c / ITERS:.0f}  {n}")
    sorted_ids, sorted_w, sorted_eids, num_valid, _ = out
    print(f"    outputs: sorted_ids {tuple(sorted_ids.shape)} expert_ids {tuple(sorted_eids.shape)} num_valid {num_valid.flatten()[:2].tolist()}")
