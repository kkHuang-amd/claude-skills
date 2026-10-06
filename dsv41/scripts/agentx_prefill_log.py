#!/usr/bin/env python3
"""Prefill-side accounting of an SGLang AgentX run from its server.log "Prefill batch" lines.

    python3 -I agentx_prefill_log.py <label>=<server.log> [...]

Reports token-weighted prefix-cache hit, new (computed) tokens per prefill batch, how much of the run went to big
misses (batches that fill a whole chunk), an estimate of the GPU seconds spent computing new prefill tokens (cold rate
from the 2026-10-06 prefix sweep: ~16 ms per 1k tokens at long context, ~21 ms at 32k -> 18 ms/1k used), and how often a
prefill ran with requests already queued.
"""
import re
import statistics as st
import sys
from datetime import datetime

PAT = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), "
                 r"#cached-token: (\d+).*?#running-req: (\d+), #queue-req: (\d+), #pending-token: (\d+)")
MS_PER_1K_NEW = 18.0


def q(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p / 100 * len(a)))] if a else 0


def main():
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        rows = []
        for line in open(path, errors="replace"):
            m = PAT.search(line)
            if m:
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                rows.append((t,) + tuple(int(x) for x in m.groups()[1:]))
        rows = [r for r in rows if r[2] > 1]  # drop the 1-token health request
        if not rows:
            print(f"== {label}: no prefill lines"); continue
        span = (rows[-1][0] - rows[0][0]).total_seconds()
        new = [r[2] for r in rows]; cached = [r[3] for r in rows]
        big = [r for r in rows if r[2] >= 16384]
        est_s = sum(new) / 1000 * MS_PER_1K_NEW / 1000
        queued = sum(1 for r in rows if r[5] > 0)
        print(f"== {label}: {len(rows)} prefill batches over {span / 60:.0f} min")
        print(f"  token hit {sum(cached) / (sum(cached) + sum(new)):.3f}; new tokens total {sum(new) / 1e6:.1f}M, "
              f"cached {sum(cached) / 1e6:.0f}M")
        print(f"  new tokens per batch p50/p90/p99/max {q(new, 50)}/{q(new, 90)}/{q(new, 99)}/{max(new)}; "
              f"batches with 0 cached: {sum(1 for c in cached if c == 0)}")
        print(f"  full-chunk (>=16384 new) batches {len(big)} = {sum(r[2] for r in big) / sum(new):.0%} of new tokens")
        print(f"  est. GPU time computing new prefill tokens {est_s:.0f} s = {est_s / span:.0%} of the window "
              f"(at {MS_PER_1K_NEW} ms/1k)")
        print(f"  prefill batches with queue-req>0: {queued / len(rows):.0%}; pending-token p50/p90 "
              f"{q([r[6] for r in rows], 50)}/{q([r[6] for r in rows], 90)}")


if __name__ == "__main__":
    main()
