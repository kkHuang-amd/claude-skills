#!/usr/bin/env python3
"""Per-point TTT (total tok/s/GPU) / P90 interactivity / TTFT p50 / ok from InferenceX agg json(s).
Usage: agentx_agg_table.py <agg_bmk.json | result.json> [...]   (list files or single-point jsons)"""
import json, sys
rows = []
for p in sys.argv[1:]:
    d = json.load(open(p))
    rows += d if isinstance(d, list) else [d]
print("| tp | conc | TTT/gpu | P90 intvty | TPOT p50 ms | TTFT p50 ms | ok/total |\n|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["tp"], r["conc"])):
    m = r["request_metrics"]; t = m["throughput"]["per_gpu"]; L = m["latency"]
    ttt = t["total_tput_tps"]
    print(f"| {r['tp']} | {r['conc']} | {ttt:,.1f} | {L['intvty']['p90']:.1f} | {L['tpot']['p50']:.3f} | "
          f"{L['ttft']['p50']:.0f} | {r['num_requests_successful']}/{r['num_requests_total']} |")
