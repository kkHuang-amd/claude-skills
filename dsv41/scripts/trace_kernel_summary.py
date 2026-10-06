#!/usr/bin/env python3
"""Kernel summary of a torch-profiler trace (one rank), in the B200_REQUEST_1006.md Deliverable 3 format.

    python3 -I trace_kernel_summary.py <trace.json[.gz]> [--steps 40] [--top 30] [--csv out.csv]

GPU (device) kernel events only. Reports the profiled window, GPU busy/idle %, ms per step (window / --steps),
per-category totals and the top-N kernels by total time. Category regexes are first-match, case-insensitive.
"""
import argparse
import collections
import gzip
import json
import re

CATS = [
    ("indexer", r"indexer|mqa_logits|top_?k|topk"),
    ("kv_compress", r"compress"),
    ("sparse_mla_attn", r"mla|sparse|attn|attention|pa_decode|flash|fmha"),
    ("moe", r"moe|expert|topk_softmax|grouped"),
    ("comm", r"all_?reduce|allgather|all_gather|reduce_scatter|cross_device|rccl|nccl|custom_ar|quick_?reduce"),
    ("engram", r"engram"),
    ("dense_gemm", r"gemm|matmul|cijk|hipblaslt|mfma|gemv|bmm|_mm_|linear"),
    ("sampling_draft", r"sampl|softmax|argmax|multinomial|accept|draft|verify"),
    ("norm_rope_elem", r"norm|rope|rotary|elementwise|vectorized|copy|cat_|fill|add|mul|silu|act|quant|cast|index"),
]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace"); p.add_argument("--steps", type=int, default=40)
    p.add_argument("--top", type=int, default=30); p.add_argument("--csv")
    a = p.parse_args()
    ev = [e for e in load(a.trace) if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not ev:
        print("no GPU kernel events"); return
    ev.sort(key=lambda e: e["ts"])
    t0, t1 = ev[0]["ts"], max(e["ts"] + e["dur"] for e in ev)
    busy = 0; cur_s = cur_e = None
    for e in ev:  # union of kernel intervals
        s, en = e["ts"], e["ts"] + e["dur"]
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, en
        else:
            cur_e = max(cur_e, en)
    busy += cur_e - cur_s
    win = t1 - t0
    by = collections.defaultdict(lambda: [0, 0.0])
    for e in ev:
        by[e["name"]][0] += 1; by[e["name"]][1] += e["dur"]
    tot = sum(v[1] for v in by.values())
    cat = collections.defaultdict(float)
    for n, (_, us) in by.items():
        c = next((c for c, rx in CATS if re.search(rx, n, re.I)), "other")
        cat[c] += us
    print(f"window {win / 1e3:.1f} ms, GPU busy {busy / win:.0%} (idle {1 - busy / win:.0%}), "
          f"{win / 1e3 / a.steps:.3f} ms/step over {a.steps} steps; kernel time {tot / 1e3:.1f} ms")
    print("category,total_ms,ms_per_step,pct")
    for c, us in sorted(cat.items(), key=lambda x: -x[1]):
        print(f"{c},{us / 1e3:.2f},{us / 1e3 / a.steps:.3f},{us / tot:.1%}")
    rows = sorted(by.items(), key=lambda x: -x[1][1])
    lines = [f"{n[:120].replace(',', ';')},{c},{us:.0f},{us / a.steps:.1f},{us / tot:.1%}" for n, (c, us) in rows]
    print(f"top {a.top}: kernel_name,calls,total_us,us_per_step,pct")
    print("\n".join(lines[:a.top]))
    if a.csv:
        open(a.csv, "w").write("kernel_name,calls,total_us,us_per_step,pct\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
