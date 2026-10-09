#!/usr/bin/env python3
"""Per-point TTT (total tok/s/GPU) / P90 interactivity / TPOT p50 / TTFT p50 / ok from InferenceX agentic jsons.
Usage: agentx_agg_table.py [--ref <agg_bmk.json>] <agg_bmk.json | result.json> [...]
  --ref: add a reference row per (tp, conc) and the delta vs it (e.g. the CI run's results_bmk agg_bmk.json).
  --row <label>: instead of the table, print one compact markdown row per point vs --ref (for append-only docs).
Latencies in the json are seconds; printed as ms (TPOT) and s (TTFT)."""
import json, sys


def load(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        rows += d if isinstance(d, list) else [d]
    return rows


def stats(r):
    m = r["request_metrics"]; L = m["latency"]
    return dict(ttt=m["throughput"]["per_gpu"]["total_tput_tps"], p90=L["intvty"]["p90"],
                tpot=L["tpot"]["p50"] * 1e3, ttft=L["ttft"]["p50"],
                ok=f"{r['num_requests_successful']}/{r['num_requests_total']}")


args = sys.argv[1:]
ref = {}
if args[:1] == ["--ref"]:
    ref = {(r["tp"], r["conc"]): stats(r) for r in load([args[1]])}
    args = args[2:]
if args[:1] == ["--row"]:
    label = args[1]
    for r in load(args[2:]):
        s, b = stats(r), ref[(r["tp"], r["conc"])]
        c = lambda x, f: f"{s[x]:{f}} / {b[x]:{f}} ({(s[x] / b[x] - 1) * 100:+.1f}%)"
        print(f"| {label} | {r['tp']} | {r['conc']} | {c('ttt', ',.1f')} | {c('p90', '.1f')} | {c('tpot', '.3f')} | "
              f"{c('ttft', '.2f')} | {s['ok']} |")
    sys.exit(0)
print("| tp | conc | run | TTT/gpu | P90 intvty | TPOT p50 ms | TTFT p50 s | ok/total |\n|---|---|---|---|---|---|---|---|")
for r in sorted(load(args), key=lambda r: (r["tp"], r["conc"])):
    s, k = stats(r), (r["tp"], r["conc"])
    print(f"| {k[0]} | {k[1]} | this | {s['ttt']:,.1f} | {s['p90']:.1f} | {s['tpot']:.3f} | {s['ttft']:.2f} | {s['ok']} |")
    if k in ref:
        b = ref[k]
        d = lambda x: f"{(s[x] / b[x] - 1) * 100:+.1f}%"
        print(f"| {k[0]} | {k[1]} | ref | {b['ttt']:,.1f} | {b['p90']:.1f} | {b['tpot']:.3f} | {b['ttft']:.2f} | {b['ok']} |")
        print(f"| | | delta | {d('ttt')} | {d('p90')} | {d('tpot')} | {d('ttft')} | |")
