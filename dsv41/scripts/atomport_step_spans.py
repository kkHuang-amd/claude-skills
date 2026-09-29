#!/usr/bin/env python3
"""Per-step GPU span from torch-profiler traces (SGLang /start_profile, GPU-only, sync-scheduled decode).
Segments kernels at idle gaps > 1 ms (the host gap between decode steps), drops the partial edge segments, and
prints per trace: kernels/step, median/mean GPU span per step (first kernel start -> last kernel end), median
host gap. Lets kernel-level A/Bs be judged without the host-side noise of end-to-end tok/s.
Usage: atomport_step_spans.py LABEL=GLOB [LABEL=GLOB ...]
"""
import glob, gzip, json, statistics, sys

for arg in sys.argv[1:]:
    label, pat = arg.split("=", 1)
    f = glob.glob(pat)[0]
    ev = json.load(gzip.open(f, "rt"))["traceEvents"]
    k = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"], key=lambda e: e["ts"])
    segs, cur, end, gaps = [], [k[0]], k[0]["ts"] + k[0]["dur"], []
    for e in k[1:]:
        if e["ts"] - end > 1000:
            segs.append((cur, end)); gaps.append(e["ts"] - end); cur = []
        cur.append(e); end = max(end, e["ts"] + e["dur"])
    segs.append((cur, end))
    full = segs[1:-1]
    spans = [(en - s[0]["ts"]) / 1e3 for s, en in full]
    print(f"{label:10s} steps={len(full)} kernels/step={statistics.median(len(s) for s, _ in full):.0f} "
          f"span median={statistics.median(spans):.3f} ms mean={statistics.mean(spans):.3f} "
          f"min={min(spans):.3f} max={max(spans):.3f} | host gap median={statistics.median(gaps[1:-1] or gaps)/1e3:.3f} ms")
