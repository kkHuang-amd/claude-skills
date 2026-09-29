#!/usr/bin/env python3
"""Print the kernel sequence of one decode step from a torch-profiler trace (GPU-only /start_profile output).

Steps are segmented like atomport_step_spans.py (idle gap > 1 ms). For step STEP (default: middle one) prints
index, start offset (us from step start), duration, and a shortened kernel name, collapsing runs of identical
names. Used to locate the DSpark draft / target verify boundary inside a step.
Usage: atomport_step_sequence.py TRACE [--step N] [--width 70]
"""
import argparse, gzip, json

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--step", type=int, default=-1); ap.add_argument("--width", type=int, default=70)
a = ap.parse_args()
ev = json.load(gzip.open(a.trace, "rt"))["traceEvents"]
k = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"], key=lambda e: e["ts"])
segs, cur, end = [], [k[0]], k[0]["ts"] + k[0]["dur"]
for e in k[1:]:
    if e["ts"] - end > 1000:
        segs.append(cur); cur = []
    cur.append(e); end = max(end, e["ts"] + e["dur"])
segs.append(cur)
full = segs[1:-1]
s = full[a.step if a.step >= 0 else len(full) // 2]
t0 = s[0]["ts"]
print(f"steps={len(full)} kernels={len(s)} span={(max(e['ts'] + e['dur'] for e in s) - t0) / 1e3:.3f} ms")
i = 0
while i < len(s):
    j = i
    while j + 1 < len(s) and s[j + 1]["name"] == s[i]["name"]:
        j += 1
    n = s[i]["name"].replace("_kernel", "")[: a.width]
    print(f"{i:5d} +{s[i]['ts'] - t0:8.1f} {sum(e['dur'] for e in s[i:j + 1]):7.1f}us x{j - i + 1:<3d} {n}")
    i = j + 1
