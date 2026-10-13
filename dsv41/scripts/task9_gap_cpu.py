#!/usr/bin/env python3
"""What the host does while the GPU idles between decode steps (torch-profiler trace with CPU activities).
GPU steps are segmented at kernel idle gaps > GAP us; for each gap, CPU events (cpu_op / user_annotation /
cuda_runtime / python) overlapping it are summed by name (overlap duration only). Prints event categories, the median
gap, and per name: total overlap per gap (us) and calls per gap, for the --top names; separately the runtime calls
(hipStreamSynchronize / hipMemcpy* / hipEventSynchronize) that END inside a gap (= the host was blocked on the GPU).
Usage: task9_gap_cpu.py TRACE [--gap 300] [--top 30]"""
import argparse, collections, gzip, json, statistics as st

ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--gap", type=float, default=300)
ap.add_argument("--top", type=int, default=30); a = ap.parse_args()
ev = [e for e in json.load(gzip.open(a.trace, "rt"))["traceEvents"] if e.get("ph") == "X"]
print("cats:", dict(collections.Counter(e.get("cat") for e in ev)))
k = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
gaps, end = [], k[0]["ts"] + k[0]["dur"]
for e in k[1:]:
    if e["ts"] - end > a.gap:
        gaps.append((end, e["ts"]))
    end = max(end, e["ts"] + e["dur"])
gaps = gaps[1:-1]
print(f"gaps>{a.gap:.0f}us: n={len(gaps)} median={st.median(g1 - g0 for g0, g1 in gaps):.0f}us")
cpu = [e for e in ev if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset")]
tot, calls, blk = collections.Counter(), collections.Counter(), collections.Counter()
for g0, g1 in gaps:
    for e in cpu:
        s, f = e["ts"], e["ts"] + e["dur"]
        if f <= g0 or s >= g1:
            continue
        n = f"{e.get('cat')}:{e['name'][:80]}"
        tot[n] += min(f, g1) - max(s, g0); calls[n] += 1
        if e.get("cat") == "cuda_runtime" and g0 < f <= g1 and any(x in e["name"] for x in ("Synchronize", "Memcpy")):
            blk[n] += f - max(s, g0)
G = len(gaps)
for n, t in tot.most_common(a.top):
    print(f"{t / G:8.1f} us/gap  {calls[n] / G:5.1f} calls  {n}")
print("blocking runtime calls ending inside gaps:")
for n, t in blk.most_common(8):
    print(f"{t / G:8.1f} us/gap  {n}")
