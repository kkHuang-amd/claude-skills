#!/usr/bin/env python3
"""Geomean MI355X / GB300 per shape (same method as fixed_seq_len/RESULTS.md: per-point ratio, then
geometric mean), for two MI355X sweeps side by side (e.g. 10-21 vs the 10-22 mono sweep), TP4 and TP8.
Usage: gb300_geomean.py MI_OLD MI_NEW [--md]"""
import glob, json, math, os, re, sys

SK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # proxy-moe/
GB = {"tp4ep1": f"{SK}/fixed_seq_len_gb300/RESULTS.md", "tp8ep1": f"{SK}/fixed_seq_len_gb300_2node/RESULTS.md"}
NG = {"tp4ep1": 4, "tp8ep1": 8}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
OLD, NEW = args


def load_mi(root, recipe):
    out = {}
    for f in glob.glob(f"{root}/{recipe}/isl*-osl*-c*/result.jsonl"):
        d = os.path.dirname(f)
        m = re.search(r"isl(\d+)-osl(\d+)-c(\d+)$", d)
        if not m or not os.path.exists(f"{d}/DONE"):
            continue
        r = [json.loads(l) for l in open(f) if l.strip()][-1]
        out[tuple(map(int, m.groups()))] = dict(ttt=r["total_throughput"] / NG[recipe], p90=1000 / r["p90_itl_ms"],
                                                itl=r["median_itl_ms"], ttft=r["median_ttft_ms"])
    return out


def load_gb(path, recipe):
    rows, cur = {}, False
    for line in open(path):
        m = re.match(r"#\s+(\d+)/(\d+)\s+(\S+)", line)
        if line.startswith("#"):
            cur = bool(m) and m.group(3) == recipe
            continue
        if cur and line.startswith("| ") and line[2].isdigit():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            rows[(int(c[0]), int(c[1]), int(c[3]))] = dict(ttt=float(c[5]), ttft=float(c[7]), itl=float(c[8]),
                                                          p90=float(c[10]))
    return rows


gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
SUBSETS = [("1024/1024", lambda k: k[:2] == (1024, 1024)), ("8192/1024", lambda k: k[:2] == (8192, 1024)),
           ("70000/300", lambda k: k[:2] == (70000, 300)), ("all 3 shapes", lambda k: True),
           ("all, CONC <= 512", lambda k: k[2] <= 512), ("all, CONC <= 16 (mono acts)", lambda k: k[2] <= 16)]
for recipe in ("tp4ep1", "tp8ep1"):
    gb = load_gb(GB[recipe], recipe)
    old, new = load_mi(OLD, recipe), load_mi(NEW, recipe)
    keys = sorted(set(gb) & set(old) & set(new))
    print(f"\n### {recipe}: MI355X / GB300 geomean, 10-21 → 10-22 mono ({len(keys)} matched points)\n")
    print("| Shape / subset | points | TTT/GPU | P90 interactivity | Median ITL | Median TTFT |")
    print("|---|---|---|---|---|---|")
    for name, sel in SUBSETS:
        ks = [k for k in keys if sel(k)]
        if not ks:
            continue
        cells = []
        for f in ("ttt", "p90", "itl", "ttft"):
            a = gm([old[k][f] / gb[k][f] for k in ks]); b = gm([new[k][f] / gb[k][f] for k in ks])
            cells.append(f"{a:.3f} → **{b:.3f}**")
        print(f"| {name} | {len(ks)} | " + " | ".join(cells) + " |")
