#!/usr/bin/env python3
"""Compare aiperf live progress blocks of AgentX runs at the same profiling time (for runs cut before the final JSON).
Usage: agentx_live_compare.py MM:SS <run.nohup> [<run.nohup> ...]   (e.g. /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/env1001_c8.nohup)
Prints per run the block closest to MM:SS: done/err, (tput_in+tput_out)/2 per GPU (TP2), out tok/s, ttft p50/p95,
intvty p50 (1/TPOT, ~ the final p50 interactivity), e2e p50.
"""
import re
import sys


def blocks(path):
    cur = None
    for line in open(path, errors="replace"):
        m = re.search(r"\[realtime (\d+):(\d+) profiling\]", line)
        if m:
            if cur:
                yield cur
            cur = {"t": int(m.group(1)) * 60 + int(m.group(2))}
            continue
        if cur is None:
            continue
        for key, pat in (
            ("tin", r"tput_in=([\d,]+)/s"),
            ("tout", r"tput_out=([\d,]+)/s"),
            ("done", r"done=([\d,]+)"),
            ("err", r"err=([\d,]+)"),
        ):
            m = re.search(pat, line)
            if m and " srv " not in line:
                cur[key] = int(m.group(1).replace(",", ""))
        for name in ("ttft", "intvty", "e2e"):
            if re.search(rf"INFO\s+{name}\s", line):
                vals = dict(re.findall(r"(p\d+)=\s*([\d,]+)", line))
                cur[name] = {k: int(v.replace(",", "")) for k, v in vals.items()}
    if cur:
        yield cur


mm, ss = sys.argv[1].split(":")
target = int(mm) * 60 + int(ss)
for path in sys.argv[2:]:
    bs = [b for b in blocks(path) if "tin" in b]
    if not bs:
        print(f"{path}: no blocks")
        continue
    b = min(bs, key=lambda b: abs(b["t"] - target))
    print(
        f"{path.rsplit('/', 1)[-1]:28s} t={b['t'] // 60}:{b['t'] % 60:02d} done={b.get('done')} err={b.get('err')} "
        f"tok/s/GPU={(b['tin'] + b['tout']) / 2:,.0f} out/s={b['tout']} "
        f"ttft p50/p95={b['ttft'].get('p50')}/{b['ttft'].get('p95')}ms intvty p50={b['intvty'].get('p50')} "
        f"e2e p50={b['e2e'].get('p50')}ms"
    )
