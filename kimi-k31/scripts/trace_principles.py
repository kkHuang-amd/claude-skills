#!/usr/bin/env python3
"""Apply optimization principles 1-4 to one rank trace's VERIFY / DRAFT steps.
Usage: trace_principles.py <trace.json.gz> [STEP=VERIFY]
  P1 per-layer kernels grouped by per-step multiplicity (layer-invariant candidates)
  P2 host-sync runtime calls per step
  P3 small-kernel census (count, time, inter-kernel gaps)
  P4 stream concurrency inside the step"""
import bisect, gzip, json, re, sys
from collections import defaultdict

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
ST = sys.argv[2] if len(sys.argv) > 2 else "VERIFY"
steps = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
               if e.get("cat") == "gpu_user_annotation" and e["name"].startswith(f"step[{ST}"))
n = len(steps); starts = [s[0] for s in steps]
def step_of(ts):
    i = bisect.bisect_right(starts, ts) - 1
    return i if i >= 0 and ts <= steps[i][1] else None

kern = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
              key=lambda e: e["ts"])
ks = [e for e in kern if step_of(e["ts"]) is not None]
span = sum(b - a for a, b in steps)
print(f"{ST}: {n} steps, mean span {span/n/1e3:.2f} ms, kernels/step {len(ks)/n:.0f}")

# P1
agg = defaultdict(lambda: [0, 0.0])
for e in ks:
    a = agg[e["name"]]; a[0] += 1; a[1] += e["dur"]
groups = defaultdict(list)
for name, (c, t) in agg.items():
    groups[round(c / n, 1)].append((t / n, c / n, name))
print("\n[P1] kernels by launches/step (layers: 24 MLA/DSA, 69 KDA, 92 MoE, 93 all)")
for mult in sorted(groups, key=lambda m: -sum(x[0] for x in groups[m])):
    g = sorted(groups[mult], reverse=True)
    print(f"  x{mult:<6} {len(g):3d} kernels {sum(x[0] for x in g)/1e3:7.3f} ms/step")
    for t, c, name in g[:6]:
        print(f"      {t:8.1f} us  {name[:100]}")

# P2
sync_pat = re.compile(r"Synchronize|MemcpyDtoH|Memcpy.*D2H|hipMemcpy$|hipMemcpyAsync|EventQuery|StreamWait", re.I)
rt = [e for e in ev if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"]
t0, t1 = steps[0][0], steps[-1][1]
win = [e for e in rt if t0 <= e["ts"] <= t1 and sync_pat.search(e["name"])]
sa = defaultdict(lambda: [0, 0.0])
for e in win:
    a = sa[e["name"]]; a[0] += 1; a[1] += e["dur"]
cp = [e for e in kern if t0 <= e["ts"] <= t1 and e.get("cat") == "gpu_memcpy"]
cpa = defaultdict(lambda: [0, 0.0])
for e in cp:
    a = cpa[e["name"]]; a[0] += 1; a[1] += e["dur"]
print(f"\n[P2] host-side sync/copy runtime calls over the {ST} window (per step)")
for k, (c, t) in sorted(sa.items(), key=lambda x: -x[1][1])[:8]:
    print(f"  {c/n:6.1f}/step {t/n:8.1f} us/step  {k}")
for k, (c, t) in sorted(cpa.items(), key=lambda x: -x[1][1])[:5]:
    print(f"  gpu {k}: {c/n:.1f}/step {t/n:.1f} us/step")

# P3
small = [e for e in ks if e["dur"] < 10]
gaps = []
for i in range(1, len(ks)):
    if step_of(ks[i]["ts"]) == step_of(ks[i-1]["ts"]):
        gaps.append(max(0.0, ks[i]["ts"] - (ks[i-1]["ts"] + ks[i-1]["dur"])))
print(f"\n[P3] small kernels (<10 us): {len(small)/n:.0f}/step, {sum(e['dur'] for e in small)/n/1e3:.2f} ms/step; "
      f"<5 us: {sum(1 for e in ks if e['dur']<5)/n:.0f}/step")
print(f"     inter-kernel gap total {sum(gaps)/n/1e3:.2f} ms/step (mean {sum(gaps)/max(len(gaps),1):.2f} us)")

# P4
streams = defaultdict(float)
for e in ks:
    streams[e.get("tid")] += e["dur"]
busy = sum(e["dur"] for e in ks)
print(f"\n[P4] streams used in {ST}: " + ", ".join(f"{s}:{t/n/1e3:.2f}ms" for s, t in sorted(streams.items(), key=lambda x: -x[1])))
print(f"     sum kernel time / span = {busy/span:.3f} (>1 means overlap)")
