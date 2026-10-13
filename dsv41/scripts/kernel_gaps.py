#!/usr/bin/env python3
"""GPU busy vs idle from a rocprofv3 kernel_trace.csv (HSA level: includes graph-replayed kernels).
    kernel_gaps.py <kernel_trace.csv> [gap_us=20] [top=15]
Per agent: window, busy (union of kernel intervals), idle = window - busy, gap histogram, and the largest gaps
with the kernels on either side (names cut to 70 chars)."""
import collections, csv, sys

path = sys.argv[1]
gap_us = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
top = int(sys.argv[3]) if len(sys.argv) > 3 else 15
by_agent = collections.defaultdict(list)
with open(path) as f:
    for r in csv.DictReader(f):
        a = r.get("Agent_Id") or r.get("Agent_Id(Handle)") or "?"
        by_agent[a].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r.get("Kernel_Name", "?")))
for a, ks in sorted(by_agent.items(), key=lambda x: -len(x[1])):
    ks.sort()
    t0, t1 = ks[0][0], max(e for _, e, _ in ks)
    busy, cur_s, cur_e, gaps = 0, ks[0][0], ks[0][1], []
    prev = ks[0]
    for s, e, n in ks[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            gaps.append((s - cur_e, prev[2], n))
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
        if e >= cur_e:
            prev = (s, e, n)
    busy += cur_e - cur_s
    win = t1 - t0
    big = [g for g in gaps if g[0] >= gap_us * 1e3]
    print(f"agent {a}: kernels {len(ks)} window {win / 1e6:.2f} ms busy {busy / 1e6:.2f} ms "
          f"({100 * busy / win:.1f}%) idle {(win - busy) / 1e6:.2f} ms; gaps>={gap_us:.0f}us: {len(big)} "
          f"sum {sum(g[0] for g in big) / 1e6:.2f} ms; all gaps sum {sum(g[0] for g in gaps) / 1e6:.2f} ms")
    hist = collections.Counter()
    for g, _, _ in gaps:
        us = g / 1e3
        b = "<5" if us < 5 else "5-20" if us < 20 else "20-100" if us < 100 else "100-500" if us < 500 else ">=500"
        hist[b] += g
    print("  gap time by size (ms): " + ", ".join(f"{k} {hist[k] / 1e6:.2f}" for k in ["<5", "5-20", "20-100", "100-500", ">=500"]))
    pairs = collections.Counter()
    for g, p, n in big:
        pairs[(p[:60], n[:60])] += g
    print(f"  top gap kinds (sum ms, before -> after):")
    for (p, n), g in pairs.most_common(top):
        print(f"    {g / 1e6:7.2f}  {p} -> {n}")
    break
