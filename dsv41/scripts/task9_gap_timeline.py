#!/usr/bin/env python3
"""Timeline of one GPU idle gap between decode steps: CPU user_annotations, cpu_ops >= --min us and blocking runtime
calls, ordered by start, offsets relative to the gap start (gap = kernel idle > --gap us; --which picks the gap, default
the middle one). Also prints the GPU kernels launched just before / after the gap.
Usage: task9_gap_timeline.py TRACE [--gap 300] [--min 15] [--which N] [--pre 3000]"""
import argparse, gzip, json

ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--gap", type=float, default=300)
ap.add_argument("--min", type=float, default=15); ap.add_argument("--which", type=int, default=-1)
ap.add_argument("--pre", type=float, default=3000, help="also show CPU events starting this many us before the gap")
a = ap.parse_args()
ev = [e for e in json.load(gzip.open(a.trace, "rt"))["traceEvents"] if e.get("ph") == "X"]
k = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
gaps, end, last = [], k[0]["ts"] + k[0]["dur"], [k[0]]
for i, e in enumerate(k[1:], 1):
    if e["ts"] - end > a.gap:
        gaps.append((end, e["ts"], i))
    end = max(end, e["ts"] + e["dur"])
g0, g1, i = gaps[a.which if a.which >= 0 else len(gaps) // 2]
print(f"gap {g1 - g0:.0f} us; kernels before: {' | '.join(x['name'][:35] for x in k[i - 3:i])}")
print(f"kernels after: {' | '.join(x['name'][:35] for x in k[i:i + 3])}")
for e in sorted(ev, key=lambda e: e["ts"]):
    c = e.get("cat")
    if c in ("kernel", "gpu_memcpy", "gpu_memset", "Trace", "gpu_user_annotation"):
        continue
    s, f = e["ts"], e["ts"] + e["dur"]
    if f < g0 - a.pre or s > g1:
        continue
    if c == "user_annotation" or e["dur"] >= a.min or ("Synchronize" in e["name"]):
        print(f"{s - g0:9.0f} {e['dur']:8.0f}  {c[:10]:10s} {e['name'][:90]}")
