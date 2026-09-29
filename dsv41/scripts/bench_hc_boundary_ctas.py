#!/usr/bin/env python3
"""Sweep CTAs-per-slice of the mHC hc_boundary_prefill_kernel<true,true> (post+combine form) at prefill M.
Checks every setting is bitwise equal to the default heuristic and reports us/call + achieved GB/s.
  HIP_VISIBLE_DEVICES=4 python3 bench_hc_boundary_ctas.py"""
import sys, torch
sys.path.insert(0, "/sgl-workspace/sglang-dsv41/python")
sys.path.insert(0, "/sgl-workspace/sglang-dsv41/test/registered/kernels/ops/layernorm")
import sglang.kernels.ops.layernorm.mhc_boundary_hip as mb
from test_hc_boundary_hip import HC, H, _params, _boundary_inputs

default_ctas = mb._hc_boundary_prefill_ctas
hc_fn, hc_scale, hc_base = _params("cuda")

def run(x, res, post_in, comb_in, pre_prev):
    res_out = torch.empty_like(res); y = torch.empty((res.shape[0], H), dtype=res.dtype, device="cuda")
    pm, ps = mb._hc_boundary_partials(x, res, post_in, comb_in, pre_prev, hc_fn, res_out, y, hc_mult=HC, prefill=True)
    return res_out, y, pm, ps

def timeit(fn, warm=5, iters=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters * 1e3

for m in (4221, 12528, 16384):
    inp = _boundary_inputs(m, m, hc_fn, hc_scale, hc_base)
    rb = (m + mb._HC_BOUNDARY_BLOCK_M - 1) // mb._HC_BOUNDARY_BLOCK_M
    ref = run(*inp)
    nbytes = sum(t.numel() * t.element_size() for t in ref) + inp[0].numel() * 2 + inp[1].numel() * 2
    line = [f"M={m:6d} default={default_ctas(rb):2d}"]
    for c in (6, 8, 12, 16, 20, 24, 32):
        mb._hc_boundary_prefill_ctas = lambda n, c=c: c
        out = run(*inp)
        same = all(torch.equal(a, b) for a, b in zip(out, ref))
        us = timeit(lambda: run(*inp))
        line.append(f"c{c}:{us:6.1f}us {nbytes/us/1e3:4.0f}GB/s{'' if same else ' DIFF'}")
    mb._hc_boundary_prefill_ctas = default_ctas
    print("  ".join(line), flush=True)
