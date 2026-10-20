"""Attribute decode all-reduce time to cross-rank arrival skew.

For the k-th all-reduce kernel on every rank, take the GPU work since the previous
all-reduce (the segment) on each rank. If skew drives AR time, then on each AR
AR_time[r] ~= base + (max_r seg[r] - seg[r]); report the fit and the implied base cost.

Usage: python3 ar_wait_attrib.py <profile_dir> [DECODE]
"""

import glob
import gzip
import json
import re
import sys

COMM = re.compile(r"cross_device|reduce_scatter|allreduce|all_reduce|quickreduce", re.I)


def per_rank(f):
    ev = [e for e in json.load(gzip.open(f, "rt"))["traceEvents"]
          if e.get("ph") == "X" and e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    ars, segs, seg = [], [], 0.0
    for e in ev:
        if COMM.search(e["name"]):
            ars.append(e["dur"])
            segs.append(seg)
            seg = 0.0
        else:
            seg += e["dur"]
    return ars, segs


def main() -> None:
    d = sys.argv[1]
    ph = sys.argv[2] if len(sys.argv) > 2 else "DECODE"
    files = sorted(glob.glob(f"{d}/**/*-TP-*-{ph}.trace.json.gz", recursive=True))
    data = [per_rank(f) for f in files]
    n = min(len(a) for a, _ in data)
    xs, ys, base = [], [], []
    for k in range(1, n):
        segs = [s[k] for _, s in data]
        m = max(segs)
        for (a, s) in data:
            xs.append(m - s[k])
            ys.append(a[k])
        base.append(min(a[k] for a, _ in data))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    slope = cov / vx
    r = cov / (vx * vy) ** 0.5
    base.sort()
    print(f"ARs/rank={n} mean AR={my:.1f}us  mean skew-wait={mx:.1f}us")
    print(f"fit AR = {my - slope * mx:.1f} + {slope:.2f}*skew  (r={r:.2f})")
    print(f"min-over-ranks AR per call: p10={base[len(base) // 10]:.1f} "
          f"p50={base[len(base) // 2]:.1f} p90={base[9 * len(base) // 10]:.1f} us")


if __name__ == "__main__":
    main()
