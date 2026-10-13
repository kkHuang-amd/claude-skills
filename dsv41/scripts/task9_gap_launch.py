#!/usr/bin/env python3
"""For the first N kernels after the middle GPU idle gap (> --gap us): the runtime call that launched each (matched by
args.correlation), its CPU start relative to the gap start, and launch->kernel-start latency. Checks CPU/GPU clock
alignment and which host call ends the gap.  Usage: task9_gap_launch.py TRACE [--gap 300] [--n 12]"""
import argparse, gzip, json

ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--gap", type=float, default=300)
ap.add_argument("--n", type=int, default=12); ap.add_argument("--which", type=int, default=-1); a = ap.parse_args()
ev = [e for e in json.load(gzip.open(a.trace, "rt"))["traceEvents"] if e.get("ph") == "X"]
rt = {e["args"]["correlation"]: e for e in ev if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy")], key=lambda e: e["ts"])
gaps, end = [], k[0]["ts"] + k[0]["dur"]
for i, e in enumerate(k[1:], 1):
    if e["ts"] - end > a.gap:
        gaps.append((end, e["ts"], i))
    end = max(end, e["ts"] + e["dur"])
g0, g1, i = gaps[a.which if a.which >= 0 else len(gaps) // 2]
print(f"gap {g1 - g0:.0f} us")
for e in k[i - 2:i + a.n]:
    r = rt.get(e.get("args", {}).get("correlation"))
    rs = f"{r['name'][:22]:22s} cpu@{r['ts'] - g0:8.0f} lat {e['ts'] - r['ts']:7.0f}" if r else "no runtime match"
    print(f"gpu@{e['ts'] - g0:8.0f} {e['dur']:6.0f}  {e['name'][:45]:45s} <- {rs}")
