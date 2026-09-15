#!/usr/bin/env python3
"""Decompose ITL from SGLang scheduler `Decode batch` lines.

    python3 decode_stats.py <server.log> [<other server.log> ...]

Why this before a torch trace: ITL is not a kernel number, it is

    ITL = step_time / accept_len
    step_time = batch_per_rank * accept_len / gen_throughput_per_rank

so a platform can lose on ITL three different ways -- it runs a bigger batch
per step, it fell out of cuda-graph replay, or its kernels are genuinely slower
at the same batch. Those need different fixes, and the scheduler log separates
them for free. Only once batch and graph status are matched does a kernel trace
attribute anything.

Reports the steady-state middle half of the run to drop ramp-up and drain.
"""
import re
import statistics as st
import sys

PAT = re.compile(
    r"Decode batch.*?#running-req: (\d+)"
    r".*?full token usage: ([\d.]+)"
    r".*?accept len: ([\d.]+)"
    r".*?cuda graph: (\w+)"
    r".*?gen throughput \(token/s\): ([\d.]+)"
)


def parse(path):
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            m = PAT.search(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                             m.group(4) == "True", float(m.group(5))))
    return rows


def report(path, rows):
    print(f"\n{path}")
    if not rows:
        print("  no `Decode batch` lines matched -- different log format or log_level too low")
        return
    mid = rows[len(rows) // 4: len(rows) * 3 // 4] or rows
    batch = [r[0] for r in mid]
    kv = [r[1] for r in mid]
    acc = [r[2] for r in mid]
    tput = [r[4] for r in mid]
    graph = sum(r[3] for r in mid) / len(mid)

    # Per-step time implied by the rank's own emission rate, then per-token ITL.
    step_ms = [1000.0 * b * a / t for b, a, t in
               ((r[0], r[2], r[4]) for r in mid) if t > 0]
    itl_ms = [s / a for s, a in zip(step_ms, acc)]

    print(f"  lines={len(rows)}  steady-state n={len(mid)}")
    for name, v in (("running-req/rank", batch), ("kv pool usage", kv),
                    ("accept len", acc), ("gen tput/rank tok/s", tput),
                    ("implied step ms", step_ms), ("implied ITL ms/token", itl_ms)):
        print(f"  {name:22s} p50={st.median(v):8.2f}  mean={st.fmean(v):8.2f}"
              f"  p90={sorted(v)[int(0.9 * len(v)) - 1]:8.2f}")
    print(f"  {'cuda graph replay':22s} {100.0 * graph:.1f}% of steps")

    # step_time(batch) curve. This is the comparison that attributes: a constant
    # offset between two curves is per-step overhead (launch/sync/interruption),
    # a slope difference is kernel efficiency. Two aggregate ITL numbers cannot
    # tell those apart, because a config that admits more requests gets a bigger
    # batch and a longer step for free.
    by_batch = {}
    for b, _kv, a, _g, t in mid:
        if t > 0:
            by_batch.setdefault(b, []).append(1000.0 * b * a / t)
    print("  step ms by running-req/rank:")
    for b in sorted(by_batch):
        v = by_batch[b]
        if len(v) >= 20:
            print(f"    batch={b:3d}  n={len(v):5d}  step_ms p50={st.median(v):7.2f}"
                  f"  per-token={st.median(v) / 3.77:6.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p, parse(p))
