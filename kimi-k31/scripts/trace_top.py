#!/usr/bin/env python3
"""Top GPU kernels in one rank trace. Usage: trace_top.py <trace.json.gz> [topN]"""
import gzip, json, sys
from collections import defaultdict

d = json.load(gzip.open(sys.argv[1]))
ev = d["traceEvents"] if isinstance(d, dict) else d
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cats = defaultdict(int)
for e in ev:
    cats[e.get("cat")] += 1
print("events", len(ev), "by cat", dict(sorted(cats.items(), key=lambda x: -x[1])[:8]))
t0 = min(e["ts"] for e in k); t1 = max(e["ts"] + e["dur"] for e in k)
busy = sum(e["dur"] for e in k)
print(f"gpu span {(t1-t0)/1e3:.1f} ms, kernel busy {busy/1e3:.1f} ms, kernels {len(k)}")
agg = defaultdict(lambda: [0, 0.0])
for e in k:
    a = agg[e["name"]]; a[0] += 1; a[1] += e["dur"]
for n, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
    print(f"{t/1e3:9.2f} ms {100*t/busy:5.1f}% n={c:6d} avg={t/c:8.1f}us  {n[:110]}")
