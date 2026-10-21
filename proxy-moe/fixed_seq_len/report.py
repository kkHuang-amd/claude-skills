#!/usr/bin/env python3
"""Proxy-MoE fixed_seq_len tables from <OUT>/<recipe>/isl*-osl*-c*/result.jsonl.
Columns follow the dsv4 fixed_seq_len report (InferenceX): TTT = total_throughput (in + out tok/s),
Interactivity = 1000 / median ITL; plus TTT/GPU and P90 interactivity (1000 / p90 ITL, dsv41 style).
Integrity warnings go to stderr. Usage: python3 report.py [OUT] [--md]"""
import glob, json, os, re, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ROOT = args[0] if args else "/shared_nfs/kk/results/proxy_24l/fixed_seq_len"
MD = "--md" in sys.argv
NGPU = {"tp8ep1": 8, "tp8ep8": 8, "tp4ep1": 4, "tp4ep4": 4}
TPE = {"tp8ep1": "8,1,1", "tp8ep8": "8,1,8", "tp4ep1": "4,1,1", "tp4ep4": "4,1,4"}
COLS = ["Input_len", "output_len", "TP,DP,EP", "Concurrency", "TTT (tok/s)", "TTT/GPU (tok/s)",
        "Median E2EL (ms)", "Median TTFT (ms)", "Median ITL (ms)", "Interactivity (tok/s/user)",
        "P90 interactivity (tok/s/user)"]

rows, warn = [], []
for d in sorted(glob.glob(os.path.join(ROOT, "*", "isl*"))):
    recipe = os.path.basename(os.path.dirname(d))
    m = re.match(r"isl(\d+)-osl(\d+)-c(\d+)$", os.path.basename(d))
    if not m or recipe not in NGPU:
        continue
    isl, osl, conc = int(m[1]), int(m[2]), int(m[3])
    f = os.path.join(d, "result.jsonl")
    if not (os.path.exists(f) and os.path.exists(os.path.join(d, "DONE"))):
        warn.append(f"{recipe} {os.path.basename(d)}: no result (see server.log / client.log)")
        continue
    r = [json.loads(l) for l in open(f) if l.strip()][-1]
    mult = 4 if isl >= 32768 else 8
    if r.get("completed") != conc * mult:
        warn.append(f"{recipe} isl{isl} c{conc}: completed {r.get('completed')} != {conc * mult}")
    for k, want in (("input_lens", isl), ("output_lens", osl)):
        v = r.get(k) or []
        if v and (min(v) != want or max(v) != want):
            warn.append(f"{recipe} isl{isl} c{conc}: {k} {min(v)}-{max(v)} != {want}")
    if any(r.get("errors") or []):
        warn.append(f"{recipe} isl{isl} c{conc}: client reported errors")
    itl, p90 = r["median_itl_ms"], r["p90_itl_ms"]
    rows.append([isl, osl, recipe, conc, r["total_throughput"], r["total_throughput"] / NGPU[recipe],
                 r["median_e2e_latency_ms"], r["median_ttft_ms"], itl, 1000.0 / itl, 1000.0 / p90])

for recipe in sorted({r[2] for r in rows}):
    for isl, osl in sorted({(r[0], r[1]) for r in rows if r[2] == recipe}):
        sel = sorted((r for r in rows if r[2] == recipe and r[0] == isl), key=lambda r: r[3])
        print(f"\n# {isl}/{osl}  {recipe}")
        if MD:
            print("| " + " | ".join(COLS) + " |\n|" + "---|" * len(COLS))
        else:
            print("\t".join(COLS))
        for r in sel:
            vals = [str(r[0]), str(r[1]), TPE[r[2]], str(r[3])] + [f"{x:.6f}" for x in r[4:]]
            print(("| " + " | ".join(vals) + " |") if MD else "\t".join(vals))
if warn:
    print("\n# WARNINGS", file=sys.stderr)
    for w in warn:
        print("#  " + w, file=sys.stderr)
