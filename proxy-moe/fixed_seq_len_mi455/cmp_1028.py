#!/usr/bin/env python3
"""MI455 overnight 2026-10-28 (h13-17): new best config vs same-night 10-23 config rerun vs the 10-23 sweep, and vs MI355X / GB300
(tp4ep1, 12 layers). TTT/GPU = total token throughput / GPUs, P90 = 1000 / p90 ITL (tok/s/user), TTFT = median ms.
Usage: python3 cmp_1028.py > RESULTS_1028.md"""
import os

from cmp_platforms import GB300, MI355, gmean, parse_report

R = "/shared_nfs/kk/results/proxy_12l"
NEW = {r: parse_report(f"{R}/fixed_seq_len_mi455_allon_h13-17_1028/REPORT.md", r) for r in ("tp8ep1", "tp8ep8")}
OLD = {r: parse_report(f"{R}/fixed_seq_len_mi455_oldcfg_h13-17_1028/REPORT.md", r) for r in ("tp8ep1", "tp8ep8")}
# 10-23 from its raw results (report.py), not RESULTS.md (whose cross-platform tables confuse parse_mi455)
D23 = {r: parse_report(f"{R}/fixed_seq_len_mi455_h13-17/REPORT.md", r) for r in ("tp8ep1", "tp8ep8")}
M355, GB = parse_report(MI355), parse_report(GB300)
SHAPES = {"1024/1024": [2, 4, 8, 16, 32, 64], "8192/1024": [2, 4, 8, 16, 32, 64], "70000/300": [2, 4, 8, 16, 32]}

print("# MI455 overnight sweep 2026-10-28 (h13-17, 12-layer proxy, TP8): new config vs old config vs 10-23, vs MI355X / GB300\n")
print("new = all-on (aiter attention, vectorized_5d KV, page 64, DDA on, tile rule, gather EP compact, ...); old = 10-23 config "
      "rerun the same night on the same node; 10-23 = the original sweep. MI355X / GB300 = tp4ep1, 12 layers (per-GPU numbers).\n")
for rec in ("tp8ep1", "tp8ep8"):
    print(f"## {rec}: TTT/GPU (tok/s) and gains\n")
    print("| shape | CONC | new | old 10-28 | 10-23 | new / old | new / 10-23 | P90 new / old | med TTFT new / old (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sh, cs in SHAPES.items():
        for c in cs:
            n, o, d = NEW[rec].get((sh, c)), OLD[rec].get((sh, c)), D23[rec].get((sh, c))
            if not (n and o and d):
                continue
            print(f"| {sh} | {c} | {n[0]:.1f} | {o[0]:.1f} | {d[0]:.1f} | {n[0] / o[0]:.2f}x | {n[0] / d[0]:.2f}x | "
                  f"{n[1]:.1f} / {o[1]:.1f} | {n[2]:.0f} / {o[2]:.0f} |")
    print()
print("## Geomean gains per shape (TTT/GPU)\n")
print("| shape | EP1 new/old | EP1 new/10-23 | EP1 old/10-23 | EP8 new/old | EP8 new/10-23 | EP8 old/10-23 |")
print("|---|---|---|---|---|---|---|")
for sh, cs in SHAPES.items():
    row = [sh]
    for rec in ("tp8ep1", "tp8ep8"):
        for a, b in ((NEW, OLD), (NEW, D23), (OLD, D23)):
            row.append(f"{gmean([a[rec][(sh, c)][0] / b[rec][(sh, c)][0] for c in cs if (sh, c) in a[rec] and (sh, c) in b[rec]]):.3f}x")
    print("| " + " | ".join(row) + " |")
print("\n## Cross-platform (best of EP1 / EP8 per point, new config)\n")
print("| shape | CONC | MI455 new best TTT/GPU | MI355X | GB300 | vs MI355X | vs GB300 | MI455 P90 | MI355X P90 | GB300 P90 | "
      "MI455 TTFT | MI355X TTFT | GB300 TTFT | 10-23 vs MI355X | 10-23 vs GB300 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
geo = {}
for sh, cs in SHAPES.items():
    for c in cs:
        k = (sh, c)
        cand = [(NEW[r][k], r) for r in NEW if k in NEW[r]]
        old23 = max(D23[r][k][0] for r in D23 if k in D23[r])
        if not cand or k not in M355 or k not in GB:
            continue
        (b, r) = max(cand, key=lambda x: x[0][0])
        m, g = M355[k], GB[k]
        geo.setdefault(sh, []).append((b[0] / m[0], b[0] / g[0], old23 / m[0], old23 / g[0]))
        print(f"| {sh} | {c} | {b[0]:.1f} ({r[-3:]}) | {m[0]:.1f} | {g[0]:.1f} | {b[0] / m[0]:.2f}x | {b[0] / g[0]:.2f}x | "
              f"{b[1]:.1f} | {m[1]:.1f} | {g[1]:.1f} | {b[2]:.0f} | {m[2]:.0f} | {g[2]:.0f} | {old23 / m[0]:.2f}x | {old23 / g[0]:.2f}x |")
print("\n| shape | geomean MI455 new / MI355X | new / GB300 | 10-23 / MI355X | 10-23 / GB300 |")
print("|---|---|---|---|---|")
for sh, v in geo.items():
    print(f"| {sh} | {gmean([x[0] for x in v]):.2f}x | {gmean([x[1] for x in v]):.2f}x | {gmean([x[2] for x in v]):.2f}x | {gmean([x[3] for x in v]):.2f}x |")
