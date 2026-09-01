#!/usr/bin/env python3
"""Build the fixed-seq-len table from runs/*/result.jsonl.
TTT = total_throughput (input+output tok/s). Interactivity = 1000/median_ITL_ms."""
import json, glob, os, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/results/dsv4-fixed_seq_len/runs"
COLS = ["Input_len","output_len","TP,DP,EP","Concurrency","TTT (tok/s)",
        "Median E2EL (ms)","Median TTFT (ms)","Median ITL (ms)","Interactivity (tok/s/user)"]

def recipe(c):
    return ("4,1,1" if c <= 32 else "4,4,1" if c <= 128 else "8,8,1")

rows, warn = [], []
for d in sorted(glob.glob(os.path.join(ROOT, "isl*"))):
    m = re.match(r"isl(\d+)-osl(\d+)-c(\d+)-", os.path.basename(d))
    if not m:
        continue
    isl, osl, conc = int(m[1]), int(m[2]), int(m[3])
    f = os.path.join(d, "result.jsonl")
    if not os.path.exists(f) or not os.path.exists(os.path.join(d, "DONE")):
        warn.append(f"{os.path.basename(d)}: no result (see server.log / client.log)")
        continue
    r = [json.loads(l) for l in open(f) if l.strip()][-1]
    itl = r["median_itl_ms"]
    # fidelity checks
    ol = r.get("output_lens") or []
    if ol and (min(ol) != osl or max(ol) != osl):
        warn.append(f"isl{isl} c{conc}: output_lens {min(ol)}-{max(ol)} != {osl}")
    il = r.get("input_lens") or []
    if il and (min(il) != isl or max(il) != isl):
        warn.append(f"isl{isl} c{conc}: input_lens {min(il)}-{max(il)} != {isl}")
    if r.get("completed") != conc * 8:
        warn.append(f"isl{isl} c{conc}: completed {r.get('completed')} != {conc*8}")
    if any(r.get("errors") or []):
        warn.append(f"isl{isl} c{conc}: client reported errors")
    rows.append([isl, osl, recipe(conc), conc, r["total_throughput"],
                 r["median_e2e_latency_ms"], r["median_ttft_ms"], itl, 1000.0 / itl])

rows.sort(key=lambda r: (r[0], r[3]))
for i, isl in enumerate(sorted({r[0] for r in rows})):
    if i:
        print()
    print(f"# {isl}/{rows[0][1]}")
    print("\t".join(COLS))
    for r in (x for x in rows if x[0] == isl):
        print("\t".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r))
if warn:
    print("\n# WARNINGS", file=sys.stderr)
    for w in warn:
        print("#  " + w, file=sys.stderr)
