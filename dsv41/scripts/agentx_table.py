#!/usr/bin/env python3
"""TTT/GPU and P90 interactivity table: our AgentX runs vs CI 37423021942 and ATOM runs.
    agentx_table.py <our_tag_prefix> [<other_tag_prefix> ...]   e.g. task9c mi355-4_task8
Points: tp2 c1/4/8/16, tp4 c1/4/8/16 (+ _r2 reruns if present). Missing data -> '-'."""
import glob, json, sys

R = "/shared_nfs/kk/results/DeepSeek-V4.1-Flash"
POINTS = [(2, 1), (2, 4), (2, 8), (2, 16), (4, 1), (4, 4), (4, 8), (4, 16)]
ci = {(d["tp"], d["conc"]): d["request_metrics"] for d in json.load(open(f"{R}/ref_ci_37423021942/agg_bmk_reconstructed.json"))}


def m(rm):
    return rm["throughput"]["per_gpu"]["total_tput_tps"], rm["latency"]["intvty"]["p90"]


def ours(tag, tp, c, suf=""):
    f = glob.glob(f"{R}/agentx/{tag}_tp{tp}_c{c}{suf}/*agentic_c{c}.json")
    return m(json.load(open(f[0]))["request_metrics"]) if f else None


def atom(run, tp, c):
    f = glob.glob(f"{R}/{run}/tp{tp}_c{c}/bmk_agentic/*")
    return m(json.load(open(f[0]))["request_metrics"]) if f else None


fmt = lambda v: "-" if v is None else f"{v[0]:,.0f} / {v[1]:.1f}"
pct = lambda a, b, i: "-" if a is None or b is None else f"{100 * (a[i] / b[i] - 1):+.1f}%"
tags = sys.argv[1:]
cols = [f"{t}" for t in tags] + ["CI 37423", "ATOM 1010", "ATOM 1007"]
print("| TP | conc | " + " | ".join(cols) + f" | {tags[0]} vs CI (TTT / P90) | vs ATOM 1010 (TTT / P90) |")
print("|" + "---|" * (len(cols) + 4))
for tp, c in POINTS:
    for suf in ("", "_r2"):
        o = [ours(t, tp, c, suf) for t in tags]
        if suf and o[0] is None:
            continue
        cr = m(ci[(tp, c)]) if (tp, c) in ci else None
        an, ao = atom("ref_atom_38056462998", tp, c), atom("ref_atom_37597985033", tp, c)
        row = [fmt(x) for x in o] + [fmt(cr), fmt(an), fmt(ao)]
        print(f"| {tp} | {c}{suf} | " + " | ".join(row) +
              f" | {pct(o[0], cr, 0)} / {pct(o[0], cr, 1)} | {pct(o[0], an, 0)} / {pct(o[0], an, 1)} |")
