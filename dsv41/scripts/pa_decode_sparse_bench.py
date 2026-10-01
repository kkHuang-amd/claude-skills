#!/usr/bin/env python3
"""Microbench aiter pa_decode_sparse at the SGLang DSV4.1 TP2 decode call shape (captured by moe_route_probe):
q [n, 32, 512] bf16, packed fp8 DSV4 cache [971, 256, 584] uint8 (145 MB -> buffer-load path), 128 indices per
query, uniform indptr, skip_reduce=True, adaptive kv_splits. --wide-cache re-lays the pool past 2 GiB (64-bit
gathers). Uses whichever aiter is first on PYTHONPATH; --save/--ref compare outputs across trees (skip_reduce=False).
  PYTHONPATH=/sgl-workspace/aiter-5833 python3 pa_decode_sparse_bench.py --n 6,12,18,24,30,36
"""
import argparse, statistics
import torch, triton

ap = argparse.ArgumentParser()
ap.add_argument("--n", default="6,12,18,24,30,36")
ap.add_argument("--topk", type=int, default=128)
ap.add_argument("--pages", type=int, default=971)
ap.add_argument("--wide-cache", action="store_true")
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--extra", action="store_true",
                help="two-loop call (most layers): + compressed extra cache [103650,128,584] (page pitch 74880 B, "
                     "~7.8 GB -> 64-bit gathers) with 512 indices per query")
ap.add_argument("--extra-topk", type=int, default=512)
ap.add_argument("--extra-pages", type=int, default=103650)
ap.add_argument("--save", default="")
ap.add_argument("--ref", default="")
a = ap.parse_args()

import aiter
from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

dev, H, D, BLOCK = "cuda", 32, 512, 256
torch.manual_seed(0)


def packed_cache(nb, block=BLOCK):
    nope, rope = D - 64, 64
    data_bytes, scale_bytes = nope + rope * 2, 8
    cache = torch.zeros(nb, block, data_bytes + scale_bytes, dtype=torch.uint8, device=dev)
    flat = cache.view(nb, block * (data_bytes + scale_bytes))
    data = flat[:, : block * data_bytes].view(nb, block, data_bytes)
    sc = flat[:, block * data_bytes:].view(nb, block, scale_bytes)
    data[:, :, :nope] = (torch.randn(nb, block, nope, device=dev) * 0.4).to(torch.float8_e4m3fn).view(torch.uint8)
    data[:, :, nope:] = (torch.randn(nb, block, rope, device=dev) * 0.4).to(torch.bfloat16).view(torch.uint8).view(
        nb, block, rope * 2)
    sc[:, :, : nope // 64] = torch.randint(124, 130, (nb, block, nope // 64), device=dev, dtype=torch.uint8)
    return cache


def big_extra_cache(nb, block=128, pitch=74880):
    # One valid page template replicated into an nb-page pool with the serving page pitch (values don't matter
    # for timing with random indices; addresses and the >2 GiB span do).
    tmpl = packed_cache(1, block)[0]
    pool = torch.empty(pitch * nb, dtype=torch.uint8, device=dev)
    view = pool.as_strided((nb, block, tmpl.shape[-1]), (pitch, tmpl.shape[-1], 1))
    view.copy_(tmpl.unsqueeze(0).expand(nb, -1, -1))
    return view


def widen(cache):
    nb, block, row = cache.shape
    pitch = max(triton.cdiv(2**31, max(1, nb - 1)), block * row)
    pitch += pitch % 2
    pool = torch.empty(pitch * (nb - 1) + block * row, dtype=cache.dtype, device=cache.device)
    v = pool.as_strided((nb, block, row), (pitch, row, 1))
    v.copy_(cache)
    return v


cache = packed_cache(a.pages)
if a.wide_cache:
    cache = widen(cache)
extra = big_extra_cache(a.extra_pages) if a.extra else None
sink = torch.randn(H, dtype=torch.float32, device=dev) * 0.1
scale = float(D) ** -0.5
outs, saved = {}, (torch.load(a.ref) if a.ref else None)
print(f"aiter={aiter.__file__.rsplit('/aiter/', 1)[0]} triton={triton.__version__} wide={a.wide_cache}")
for n in [int(x) for x in a.n.split(",")]:
    g = torch.Generator(device=dev).manual_seed(n)
    q = (torch.randn(n, H, D, device=dev, generator=g) * 0.125).to(torch.bfloat16)
    idx = torch.randint(0, a.pages * BLOCK, (n * a.topk,), device=dev, dtype=torch.int32, generator=g)
    indptr = torch.arange(0, n * a.topk + 1, a.topk, dtype=torch.int32, device=dev)
    kw = {}
    if a.extra:
        kw = dict(extra_cache=extra,
                  extra_indices=torch.randint(0, a.extra_pages * 128, (n * a.extra_topk,), device=dev,
                                              dtype=torch.int32, generator=g),
                  extra_indptr=torch.arange(0, n * a.extra_topk + 1, a.extra_topk, dtype=torch.int32, device=dev))
    fn = lambda: pa_decode_sparse(q, cache, idx, indptr, sink, scale, skip_reduce=True, **kw)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    gph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(gph, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    ts = []
    for _ in range(a.iters):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record(); gph.replay(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1e3)
    out = pa_decode_sparse(q, cache, idx, indptr, sink, scale, skip_reduce=False, **kw).float().cpu()
    outs[n] = out
    err = ""
    if saved is not None and n in saved:
        err = f"  max|diff| vs ref {float((out - saved[n]).abs().max()):.2e}"
    print(f"n={n:3d}  {statistics.median(ts):7.2f} us{err}", flush=True)
if a.save:
    torch.save(outs, a.save)
