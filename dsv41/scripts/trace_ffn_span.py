#!/usr/bin/env python3
"""Per-layer FFN span of a DSV4.1 decode torch-profiler trace (one rank): wall time from the start of the attention-output
all-reduce to the end of the MoE all-reduce, i.e. the span the mono FFN launch replaces (MONO_KERNEL_mi355-4.md P0.3).

    python3 -I trace_ffn_span.py <trace.json[.gz]> [--ar all_reduce_mhc_post] [--moe mfma_moe1]

A span = two consecutive AR kernels with a routed-MoE kernel between them. Reports count, median / p10 / p90 span (us),
median sum of kernel time inside the span, median kernel count, and the median per-kernel time by name.
"""
import argparse
import collections
import gzip
import json
import statistics


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--ar", default="all_reduce_mhc_post")
    ap.add_argument("--moe", default="mfma_moe1")
    a = ap.parse_args()
    ev = [e for e in load(a.trace)["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy")]
    ev.sort(key=lambda e: e["ts"])
    ars = [i for i, e in enumerate(ev) if a.ar in e["name"]]
    spans, sums, counts, per = [], [], [], collections.defaultdict(list)
    for i0, i1 in zip(ars, ars[1:]):
        inner = ev[i0 : i1 + 1]
        if not any(a.moe in e["name"] for e in inner):
            continue
        spans.append(ev[i1]["ts"] + ev[i1]["dur"] - ev[i0]["ts"])
        sums.append(sum(e["dur"] for e in inner))
        counts.append(len(inner))
        seen = collections.Counter()
        for e in inner:
            seen[e["name"]] += e["dur"]
        for n, d in seen.items():
            per[n].append(d)
    if not spans:
        print("no spans found")
        return
    q = statistics.quantiles(spans, n=10)
    print(f"spans={len(spans)} median_us={statistics.median(spans):.1f} p10={q[0]:.1f} p90={q[-1]:.1f} "
          f"kernel_sum_median_us={statistics.median(sums):.1f} kernels_median={statistics.median(counts):.0f}")
    rows = sorted(((statistics.median(v), len(v), n) for n, v in per.items()), reverse=True)
    for med, cnt, n in rows[:20]:
        print(f"  {med:7.1f} us  in {cnt:4d} spans  {n[:110]}")


if __name__ == "__main__":
    main()
