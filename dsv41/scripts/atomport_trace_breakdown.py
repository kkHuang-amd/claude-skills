#!/usr/bin/env python3
"""Summarize a torch-profiler trace (SGLang /start_profile output, *.trace.json[.gz]) for decode attribution.

Prints: window wall time, GPU busy % (union of kernel intervals / window), kernel count, and top kernels by total
time plus a coarse per-module grouping (regex on kernel name). Divide by the step count you profiled for per-step.
Usage: atomport_trace_breakdown.py TRACE [--steps N] [--top 25]
"""
import argparse, gzip, json, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--steps", type=int, default=1); ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()
op = gzip.open if a.trace.endswith(".gz") else open
ev = json.load(op(a.trace, "rt"))["traceEvents"]
k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
if not k:
    raise SystemExit("no GPU kernel events in trace")
k.sort(key=lambda e: e["ts"])
t0, t1 = k[0]["ts"], max(e["ts"] + e["dur"] for e in k)
busy, cur_s, cur_e = 0.0, None, None
for e in k:
    s, en = e["ts"], e["ts"] + e["dur"]
    if cur_e is None or s > cur_e:
        if cur_e is not None: busy += cur_e - cur_s
        cur_s, cur_e = s, en
    else:
        cur_e = max(cur_e, en)
busy += cur_e - cur_s
wall = t1 - t0
S = a.steps
print(f"window {wall/1e3:.2f} ms, GPU busy {100*busy/wall:.1f}%, kernels {len(k)} "
      f"| per step (/{S}): wall {wall/1e3/S:.3f} ms, busy {busy/1e3/S:.3f} ms, kernels {len(k)/S:.0f}")

gaps, end = [], k[0]["ts"] + k[0]["dur"]
for e in k[1:]:
    if e["ts"] > end:
        gaps.append(e["ts"] - end)
    end = max(end, e["ts"] + e["dur"])
print("-- idle gaps between kernels (count/step, ms/step)")
for lo, hi in ((0, 5), (5, 20), (20, 100), (100, 1000), (1000, 1e12)):
    g = [x for x in gaps if lo <= x < hi]
    print(f"  [{lo:>5},{hi if hi < 1e12 else 'inf':>5}) us: {len(g)/S:7.1f} {sum(g)/1e3/S:7.3f}")

GROUPS = [
    ("allreduce/comm", r"allreduce|all_reduce|cross_device|nccl|rccl|reduce_scatter|allgather"),
    ("moe", r"moe|fmoe|expert|topk_softmax|grouped|biased_grouped|a8w4|a4w4|flydsl_gemm2|stage[12]"),
    ("indexer/topk", r"mqa_logits|indexer|topk|block_maxima|candidate|block_scores|paged_logits"),
    ("sparse attn", r"sparse|paged_decode|decode_attn|mla|flash|attn|opus|unified_kv"),
    ("mhc", r"mhc|hyper|sinkhorn"),
    ("engram", r"engram|hash"),
    ("norm/rope/quant", r"rmsnorm|rms_norm|norm|rope|rotary|quant|mxfp8|fp8"),
    ("gemm", r"gemm|gemv|matmul|cijk|Cijk|mfma|hipblaslt|wvSplit|bf16"),
    ("sampling/spec", r"sample|argmax|softmax|accept|verify|dspark|draft"),
    ("copy/elementwise", r"elementwise|copy|memcpy|memset|fill|cat|index|scatter|gather|vectorized"),
]
by_name, by_grp = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
for e in k:
    n = e["name"]
    by_name[n][0] += e["dur"]; by_name[n][1] += 1
    g = next((g for g, p in GROUPS if re.search(p, n, re.I)), "other")
    by_grp[g][0] += e["dur"]; by_grp[g][1] += 1
print("\n-- by group (ms/step, launches/step, % of busy-sum)")
tot = sum(v[0] for v in by_grp.values())
for g, (d, c) in sorted(by_grp.items(), key=lambda x: -x[1][0]):
    print(f"{g:18s} {d/1e3/S:8.3f} {c/S:7.0f} {100*d/tot:6.1f}%")
print(f"\n-- top {a.top} kernels (ms/step, launches/step)")
for n, (d, c) in sorted(by_name.items(), key=lambda x: -x[1][0])[:a.top]:
    print(f"{d/1e3/S:8.3f} {c/S:6.1f}  {n[:130]}")
