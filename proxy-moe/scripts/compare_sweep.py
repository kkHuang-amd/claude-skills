#!/usr/bin/env python3
"""Compare fixed_seq_len sweeps point by point: mono (all points) vs base (same night, CONC <= 16)
and vs an older full sweep. Usage: compare_sweep.py MONO BASE OLD [--md]"""
import glob, json, os, re, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MONO, BASE, OLD = args
MD = "--md" in sys.argv
NGPU = {"tp8ep1": 8, "tp4ep1": 4}


def load(root):
    out = {}
    for d in glob.glob(os.path.join(root, "*", "isl*")):
        recipe = os.path.basename(os.path.dirname(d))
        m = re.match(r"isl(\d+)-osl(\d+)-c(\d+)$", os.path.basename(d))
        f = os.path.join(d, "result.jsonl")
        if not m or recipe not in NGPU or not os.path.exists(os.path.join(d, "DONE")) or not os.path.exists(f):
            continue
        r = [json.loads(l) for l in open(f) if l.strip()][-1]
        out[(recipe, int(m[1]), int(m[2]), int(m[3]))] = (
            r["total_throughput"] / NGPU[recipe], r["median_itl_ms"], r["median_ttft_ms"])
    return out


mono, base, old = load(MONO), load(BASE), load(OLD)
pct = lambda a, b: f"{(a / b - 1) * 100:+.1f}%"
hdr = ["recipe", "ISL/OSL", "CONC", "TTT/GPU mono", "vs base", "vs 10-21", "ITL ms mono", "vs base", "vs 10-21",
       "TTFT ms mono", "vs base", "vs 10-21"]
print(("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr)) if MD else "\t".join(hdr))
for k in sorted(mono, key=lambda k: (k[0] != "tp8ep1", k[1], k[3])):
    t, itl, ttft = mono[k]
    b, o = base.get(k), old.get(k)
    row = [k[0], f"{k[1]}/{k[2]}", str(k[3]),
           f"{t:.1f}", pct(t, b[0]) if b else "", pct(t, o[0]) if o else "",
           f"{itl:.2f}", pct(itl, b[1]) if b else "", pct(itl, o[1]) if o else "",
           f"{ttft:.0f}", pct(ttft, b[2]) if b else "", pct(ttft, o[2]) if o else ""]
    print(("| " + " | ".join(row) + " |") if MD else "\t".join(row))
missing = sorted(set(old) - set(mono))
if missing:
    print("missing in mono:", missing, file=sys.stderr)
