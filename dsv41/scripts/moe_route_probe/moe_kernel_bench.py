#!/usr/bin/env python3
"""MoE kernel microbench at IDENTICAL, controlled routing (DSV4.1-Flash TP2 shapes: hidden 5120, inter/rank 1152).

Runs aiter.fused_moe.fused_moe (whichever aiter is on the path -> run once in our env, once in the ATOM chroot) with
random MXFP4 weights (scales = 1.0) and topk_ids drawn so that the T tokens touch exactly D distinct routed experts.
Configs (--configs):
  ours   : E385 topk7 (6 routed + fused shared id 384), gate_mode interleave (-> a8w4 on gfx950)
  atom   : E384 topk6, gate_mode separated (-> a4w4)
  sep385 : E385 topk7 (shared fused), gate_mode separated (a4w4 with our expert layout)
Timing: CUDA-graph replay of one call (falls back to eager loop), median over --iters.
  python3 moe_kernel_bench.py --points 6:7,6:16,24:7,24:37,24:48,36:58,36:80 --configs ours,atom,sep385
"""
import argparse, os, random, statistics
import torch

os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")  # SGLang server setting: fp8 act at every M (a8w4 rows)

ap = argparse.ArgumentParser()
ap.add_argument("--points", default="6:7,6:16,24:7,24:37,24:48,36:58,36:80", help="T:D pairs")
ap.add_argument("--configs", default="ours,atom,sep385")
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

dev = torch.device("cuda")
H, I = 5120, 1152
CFG = {"ours": (385, 7, "interleave"), "atom": (384, 6, "separated"), "sep385": (385, 7, "separated")}
SCALE_DT = getattr(torch, "float8_e8m0fnu", torch.uint8)


def weights(E):
    w1 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
    s1 = torch.full((E * 2 * I, H // 32), 127, dtype=torch.uint8, device=dev).view(SCALE_DT)
    s2_cols = (I // 32 + 7) // 8 * 8  # serving layout pads 36 -> 40 (w2_scale [E*5120, 40])
    s2 = torch.full((E * H, s2_cols), 127, dtype=torch.uint8, device=dev).view(SCALE_DT)
    w1.is_shuffled = True  # both engines preshuffle; aiter picks the FlyDSL path only when this is set
    w2.is_shuffled = True
    return w1, w2, s1, s2


def routing(T, D, topk, E, rng):
    routed_k = topk - 1 if E % 2 else topk
    experts = rng.sample(range(384), D)
    rows, used = [], set()
    for t in range(T):
        # cover all D experts across the batch, then fill randomly from the set
        need = [e for e in experts if e not in used][:routed_k]
        rest = rng.sample([e for e in experts if e not in need], routed_k - len(need))
        row = need + rest
        used.update(row)
        rows.append(row + ([384] if E % 2 else []))
    ids = torch.tensor(rows, dtype=torch.int32, device=dev)
    w = torch.full((T, topk), 1.0 / topk, dtype=torch.float32, device=dev)
    return ids, w, len(used)


def bench(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        run, mode = g.replay, "graph"
    except Exception as e:
        run, mode = fn, f"eager({type(e).__name__})"
    times = []
    for _ in range(a.iters):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record(); run(); en.record(); torch.cuda.synchronize()
        times.append(st.elapsed_time(en) * 1e3)
    return statistics.median(times), mode


W = {}
for name in a.configs.split(","):
    E, topk, gm = CFG[name]
    if E not in W:
        W[E] = weights(E)
    w1, w2, s1, s2 = W[E]
    for pt in a.points.split(","):
        T, D = map(int, pt.split(":"))
        rng = random.Random(a.seed * 1000 + T * 100 + D)
        ids, tw, got = routing(T, D, topk, E, rng)
        x = (torch.randn(T, H, device=dev) * 0.1).to(torch.bfloat16)
        fn = lambda: fused_moe(x, w1, w2, tw, ids, activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                               w1_scale=s1, w2_scale=s2, gate_mode=gm, swiglu_limit=10.0)
        try:
            us, mode = bench(fn)
            print(f"{name:7s} E{E} k{topk} {gm:10s} T={T:3d} D={got:3d}  {us:7.1f} us  [{mode}]", flush=True)
        except Exception as e:
            print(f"{name:7s} E{E} k{topk} {gm:10s} T={T:3d} D={D:3d}  ERROR {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
