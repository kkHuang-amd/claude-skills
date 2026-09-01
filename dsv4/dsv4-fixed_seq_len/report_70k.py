#!/usr/bin/env python3
"""Rebuild the 70000/300 table from runs-70k/*/*.jsonl.
Interactivity = 1000/median_ITL_ms. TTT = total_throughput (in+out tok/s)."""
import json, glob, re, os, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs-70k")
CFG = {"tp8_chunk32768": "8,1,1", "tp8dp8_chunk16384_nodelayer": "8,8,1"}
COLS = ["Input_len","output_len","TP,DP,EP","Concurrency","TTT (tok/s)","Median E2EL (ms)",
        "Median TTFT (ms)","Median ITL (ms)","Interactivity (tok/s/user)"]
rows, integ = [], []
for d, tpe in CFG.items():
    for f in sorted(glob.glob(os.path.join(ROOT, d, "*.jsonl")),
                    key=lambda p: int(re.search(r"_c(\d+)\.jsonl", p)[1])):
        c = int(re.search(r"_c(\d+)\.jsonl", f)[1])
        r = [json.loads(l) for l in open(f) if l.strip()][-1]
        itl = r["median_itl_ms"]
        rows.append([70000, 300, tpe, c, r["total_throughput"], r["median_e2e_latency_ms"],
                     r["median_ttft_ms"], itl, 1000.0 / itl])
        integ.append(f"  c{c:<3} completed={r['completed']}/{c*4} "
                     f"errors={sum(1 for x in (r.get('errors') or []) if x)}  [{d}]")
print("# 70000/300")
print("\t".join(COLS))
for r in rows:
    print("\t".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r))
print("\n# integrity"); print("\n".join(integ))
