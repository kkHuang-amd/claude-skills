"""Standalone timing of aiter's FlyDSL MoE gather-reduce (B in MOE_KERNEL_DESIGN_h13-17_1028.md).

out[t] = sum_k w[t,k] * grouped[row(t,k)], grouped (E, max_m, H) bf16. Shapes: EP1 (T=8192, top-16 of 512, H=16384) and
EP8 (one rank: T=8192, 2 local routes per token on average over 64 experts; modelled as top-2 of 64).
Usage: python3 gather_reduce_bench.py [vec ...]   (vec_dwords override; default: the wrapper's own choice)
"""
import sys

import torch

import aiter.ops.flydsl.grouped_moe_gfx1250 as gm


def make(T, topk, E, H, max_m, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(T, E, generator=g)
    ids = scores.topk(topk, dim=1).indices  # distinct experts per token
    counts = torch.zeros(E, dtype=torch.long)
    rows = torch.empty(T, topk, dtype=torch.int32)
    for t in range(T):  # row = expert * max_m + position within the expert
        for k in range(topk):
            e = int(ids[t, k])
            rows[t, k] = e * max_m + int(counts[e])
            counts[e] += 1
    assert int(counts.max()) <= max_m, int(counts.max())
    grouped = torch.randn(E, max_m, H, dtype=torch.bfloat16, device="cuda")
    w = torch.rand(T, topk, dtype=torch.float32, device="cuda")
    return grouped, rows.cuda(), w


def make_ep(T, topk, E_total, E_local, H, max_m, seed=0):
    """EP rank view: topk slots per token over E_total experts; routes to experts outside [0, E_local) are dropped
    (row = DROPPED_ROUTE_ROW = -1, weight 0), as the EP route kernel produces them."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.rand(T, E_total, generator=g).topk(topk, dim=1).indices
    counts = torch.zeros(E_local, dtype=torch.long)
    rows = torch.full((T, topk), -1, dtype=torch.int32)
    w = torch.zeros(T, topk, dtype=torch.float32)
    for t in range(T):
        for k in range(topk):
            e = int(ids[t, k])
            if e < E_local:
                rows[t, k] = e * max_m + int(counts[e])
                counts[e] += 1
                w[t, k] = float(torch.rand(1, generator=g))
    grouped = torch.randn(E_local, max_m, H, dtype=torch.bfloat16, device="cuda")
    return grouped, rows.cuda(), w.cuda()


def bench(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    vecs = [int(v) or None for v in sys.argv[1:]] or [None]  # 0 = the wrapper default
    orig = gm._choose_gather_reduce_vec
    Ts = [int(x) for x in __import__("os").environ.get("GR_BENCH_T", "8192").split(",")]
    cases = [("EP1", 8192, 16, 512, 16384, 384, None)] if 8192 in Ts else []
    cases += [(f"EP8-real T={t}", t, 16, 512, 16384, 384, 64) for t in Ts]
    for name, T, topk, E, H, max_m, e_local in cases:
        grouped, rows, w = make(T, topk, E, H, max_m) if e_local is None else make_ep(T, topk, E, e_local, H, max_m)
        out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
        ref = None
        for v in vecs:
            gm._choose_gather_reduce_vec = orig if v is None else (lambda *_a, _v=v: _v)
            nvt = torch.tensor([T], dtype=torch.int32, device="cuda") if e_local is not None else None  # EP path
            fn = lambda: gm.flydsl_moe_gather_reduce(grouped, rows, w, out=out, num_valid_tokens=nvt)  # noqa: E731
            ms = bench(fn)
            if ref is None:  # correctness vs torch (f32 accumulate)
                r = rows.view(-1).long()
                g = grouped.view(-1, H)[r.clamp(min=0)].view(T, topk, H).float()
                g = g * (rows >= 0).view(T, topk, 1)
                ref = (g * w.unsqueeze(-1)).sum(1).to(torch.bfloat16)
            err = (out.float() - ref.float()).abs().max().item()
            nbytes = int((rows >= 0).sum()) * H * 2 + T * H * 2  # live routes only
            print(f"{name} vec={v if v else orig(T, H)}: {ms:.3f} ms  {nbytes / ms / 1e9:.2f} TB/s  max|err| {err:.3g}")
    gm._choose_gather_reduce_vec = orig


if __name__ == "__main__":
    main()
