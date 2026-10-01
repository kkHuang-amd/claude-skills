#!/usr/bin/env python3
"""Per-step kernel time vs decode batch size, SGLang vs ATOM, from decode_bs_sweep.py --profile-dir traces.

Absolute cross-engine kernel times are confounded by profiler overhead (ATOM profiles CPU+GPU and launches ~1.6x
more kernels/step), so the useful signal is the GROWTH from bs1 within each engine (same kernel set, same overhead).
SGLang steps = --sgl-steps (the /start_profile num_steps); ATOM steps = count of a once-per-spec-step kernel.
  python3 bs_scaling_compare.py [--bs 1,2,4,6] [--top 15]
"""
import argparse, glob, gzip, json, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--sgl", default="/shared_nfs/kk/dsv41/atomport/prof_bsweep/prof_sgl_ctx65536_bs{bs}/*TP-0.trace.json.gz")
ap.add_argument("--atom", default="/shared_nfs/kk/atom_run/prof_bsweep/bs{bs}/*.json.gz")
ap.add_argument("--sgl-steps", type=int, default=40)
ap.add_argument("--atom-step-kernel", default="rejection_synthetic_sample_kernel")
ap.add_argument("--bs", default="1,2,4,6")
ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()

GROUPS = [
    ("allreduce/comm", r"allreduce|all_reduce|cross_device|nccl|rccl|reduce_scatter|allgather|ar_ll"),
    ("moe", r"moe|fmoe|expert|topk_gating|topk_softmax|grouped|a8w4|a4w4|flydsl_gemm2|stage[12]|act_and_mul|router"),
    ("indexer/topk", r"mqa_logits|indexer|topk|block_maxima|candidate|block_scores|paged_logits"),
    ("sparse attn", r"sparse|paged_decode|decode_attn|mla|flash|attn|opus|unified_kv"),
    ("mhc", r"mhc|hyper|sinkhorn|hc_boundary|pre_mix|collapse"),
    ("norm/rope/quant", r"rmsnorm|rms_norm|norm|rope|rotary|quant|mxfp8|fp8"),
    ("gemm", r"gemm|gemv|matmul|cijk|mfma|hipblaslt|wvsplit"),
    ("sampling/spec", r"sample|argmax|softmax|accept|verify|dspark|draft"),
]


def group(n):
    s = n.lower()
    for g, p in GROUPS:
        if re.search(p, s):
            return g
    return "other"


def family(n):
    n = re.sub(r"^void ", "", n)
    n = re.sub(r"[<(].*", "", n)
    return n[:60]


def load(path, steps=None, step_kernel=None):
    ev = json.load(gzip.open(sorted(glob.glob(path))[0], "rt"))["traceEvents"]
    k = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
    if steps is None:
        steps = sum(1 for e in k if step_kernel in e["name"])
    busy, cs, ce = 0.0, None, None
    for e in k:
        s, en = e["ts"], e["ts"] + e["dur"]
        if ce is None or s > ce:
            if ce is not None:
                busy += ce - cs
            cs, ce = s, en
        else:
            ce = max(ce, en)
    busy += ce - cs
    wall = max(e["ts"] + e["dur"] for e in k) - k[0]["ts"]
    grp, fam = defaultdict(float), defaultdict(float)
    for e in k:
        grp[group(e["name"])] += e["dur"] / 1e3 / steps
        fam[family(e["name"])] += e["dur"] / 1e3 / steps
    return dict(steps=steps, wall=wall / 1e3 / steps, busy=busy / 1e3 / steps, nk=len(k) / steps, grp=grp, fam=fam)


bss = [int(x) for x in a.bs.split(",")]
R = {("S", b): load(a.sgl.format(bs=b), steps=a.sgl_steps) for b in bss}
R.update({("A", b): load(a.atom.format(bs=b), step_kernel=a.atom_step_kernel) for b in bss})
lo, hi = bss[0], bss[-1]
hdr = "".join(f"  S{b:<5d}" for b in bss) + "".join(f"  A{b:<5d}" for b in bss)
print(f"{'ms/step':18s}{hdr}   dS{lo}->{hi}  dA{lo}->{hi}")
for key in ("steps", "nk", "wall", "busy"):
    v = [R[("S", b)][key] for b in bss] + [R[("A", b)][key] for b in bss]
    d = f"{R[('S', hi)][key] - R[('S', lo)][key]:+8.2f}  {R[('A', hi)][key] - R[('A', lo)][key]:+8.2f}" if key != "steps" else ""
    print(f"{key:18s}" + "".join(f"{x:8.2f}" for x in v) + "   " + d)
for g in [g for g, _ in GROUPS] + ["other"]:
    v = [R[("S", b)]["grp"][g] for b in bss] + [R[("A", b)]["grp"][g] for b in bss]
    print(f"{g:18s}" + "".join(f"{x:8.2f}" for x in v) +
          f"   {v[len(bss) - 1] - v[0]:+8.2f}  {v[-1] - v[len(bss)]:+8.2f}")
for eng in ("S", "A"):
    fl, fh = R[(eng, lo)]["fam"], R[(eng, hi)]["fam"]
    top = sorted(set(fl) | set(fh), key=lambda f: -(fh.get(f, 0) - fl.get(f, 0)))[:a.top]
    print(f"-- {eng}: top kernel growth bs{lo}->bs{hi} (ms/step: bs{lo} -> bs{hi}, delta)")
    for f in top:
        print(f"   {fl.get(f, 0):6.3f} -> {fh.get(f, 0):6.3f}  {fh.get(f, 0) - fl.get(f, 0):+6.3f}  {f}")
