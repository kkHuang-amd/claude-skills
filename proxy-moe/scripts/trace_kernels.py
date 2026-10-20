"""Aggregate GPU kernel time from a torch profiler chrome trace (.json.gz).

Usage: python3 trace_kernels.py <trace.json.gz> [--top 25] [--steps N]
Prints per-kernel total/count/share and a coarse category breakdown.
"""

import argparse
import gzip
import json
import re
from collections import defaultdict

CATS = [
    ("moe", r"moe|fused_moe|flydsl.*(gemm|mm)|mxfp4_moe|ck_moe|topk|sort"),
    ("quant", r"quant|mxfp4|fp4|e8m0"),
    ("attn", r"attn|attention|paged|mha|fmha|flash|decode_|kv_cache|rope|rotary"),
    ("comm", r"all_reduce|allreduce|nccl|rccl|cross_device|all_gather|reduce_scatter"),
    ("gemm", r"gemm|cijk|Cijk|matmul|hipblaslt|wvSplitK|_mm_"),
    ("norm", r"norm|rms"),
]


def cat(name: str) -> str:
    for c, pat in CATS:
        if re.search(pat, name, re.I):
            return c
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--steps", type=int, default=1, help="divide totals by this")
    a = ap.parse_args()
    with gzip.open(a.trace, "rt") as f:
        ev = json.load(f)["traceEvents"]
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            tot[e["name"]] += e.get("dur", 0)
            cnt[e["name"]] += 1
    all_us = sum(tot.values())
    s = a.steps
    print(f"total GPU kernel time: {all_us / s / 1000:.2f} ms/step  kernels={len(tot)}")
    cats = defaultdict(float)
    for n, t in tot.items():
        cats[cat(n)] += t
    print("by category: " + "  ".join(
        f"{c}={t / s / 1000:.2f}ms({100 * t / all_us:.1f}%)"
        for c, t in sorted(cats.items(), key=lambda x: -x[1])))
    for n, t in sorted(tot.items(), key=lambda x: -x[1])[: a.top]:
        print(f"{100 * t / all_us:5.1f}% {t / s / 1000:8.3f}ms n={cnt[n] // s:5d} [{cat(n)}] {n[:110]}")


if __name__ == "__main__":
    main()
