#!/usr/bin/env python3
"""MI455 (h13-17, TP8, GFXCLK locked 1000 MHz, 10-29) vs MI355X / GB300 (tp4ep1), 12-layer proxy, per-GPU numbers.
MI455 = best of tp8ep1 / tp8ep8 per point by TTT/GPU (also shown: 10-28 DPM best). TTT/GPU = total token throughput / GPU,
P90 = 1000 / p90 ITL (tok/s/user), TTFT = median ms. Usage: python3 cmp_platforms_lock1000_1029.py"""
from cmp_platforms import GB300, MI355, gmean, parse_report

R = "/shared_nfs/kk/results/proxy_12l"
def best(path):
    a, b = parse_report(path, "tp8ep1"), parse_report(path, "tp8ep8")
    return {k: max((a[k] + ("EP1",), b[k] + ("EP8",)) if k in b else (a[k] + ("EP1",),), key=lambda x: x[0]) for k in a}
L, D = best(f"{R}/sweep_lock1000_h13-17_1029/REPORT.md"), best(f"{R}/fixed_seq_len_mi455_allon_h13-17_1028/REPORT.md")
M, G = parse_report(MI355), parse_report(GB300)
print("| shape | c | MI455 lock1000 TTT/GPU (recipe) | MI355X | GB300 | lock / MI355X | lock / GB300 | DPM 10-28 / MI355X | DPM / GB300 "
      "| P90 MI455 / MI355X / GB300 | TTFT MI455 / MI355X / GB300 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
per = {}
for k in sorted(L, key=lambda k: (int(k[0].split('/')[0]), k[1])):
    if k not in M or k not in G: continue
    l, d, m, g = L[k], D.get(k), M[k], G[k]
    per.setdefault(k[0], []).append((l[0] / m[0], l[0] / g[0], d[0] / m[0] if d else None, d[0] / g[0] if d else None))
    print(f"| {k[0]} | {k[1]} | {l[0]:.1f} ({l[3]}) | {m[0]:.1f} | {g[0]:.1f} | {l[0]/m[0]:.2f}x | {l[0]/g[0]:.2f}x "
          f"| {d[0]/m[0]:.2f}x | {d[0]/g[0]:.2f}x | {l[1]:.0f} / {m[1]:.0f} / {g[1]:.0f} | {l[2]:.0f} / {m[2]:.0f} / {g[2]:.0f} |")
print("\n| shape | geomean lock1000 / MI355X | lock1000 / GB300 | DPM 10-28 / MI355X | DPM 10-28 / GB300 |")
print("|---|---|---|---|---|")
allv = []
for sh, v in per.items():
    allv += v
    print(f"| {sh} | {gmean([x[0] for x in v]):.2f}x | {gmean([x[1] for x in v]):.2f}x | {gmean([x[2] for x in v]):.2f}x | {gmean([x[3] for x in v]):.2f}x |")
print(f"| all | {gmean([x[0] for x in allv]):.2f}x | {gmean([x[1] for x in allv]):.2f}x | {gmean([x[2] for x in allv]):.2f}x | {gmean([x[3] for x in allv]):.2f}x |")
