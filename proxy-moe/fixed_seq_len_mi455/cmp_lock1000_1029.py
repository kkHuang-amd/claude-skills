#!/usr/bin/env python3
"""h13-17 2026-10-29: full sweep with all-socket GFXCLK locked at 1000 MHz vs the 10-28 DPM sweep (same MODE=new config).
TTT/GPU = total token throughput / GPU, P90 = 1000 / p90 ITL (tok/s/user), TTFT = median ms. Usage: python3 cmp_lock1000_1029.py"""
from cmp_platforms import gmean, parse_report

R = "/shared_nfs/kk/results/proxy_12l"
for r in ("tp8ep1", "tp8ep8"):
    old = parse_report(f"{R}/fixed_seq_len_mi455_allon_h13-17_1028/REPORT.md", r)
    new = parse_report(f"{R}/sweep_lock1000_h13-17_1029/REPORT.md", r)
    print(f"\n### {r.upper()}: 10-28 DPM -> 10-29 LOCK1000\n")
    print("| shape | c | TTT/GPU | delta | P90 tok/s/user | delta | TTFT med ms | delta |")
    print("|---|---|---|---|---|---|---|---|")
    rt, rp, rf = [], [], []
    for k in sorted(new, key=lambda k: (int(k[0].split('/')[0]), k[1])):
        if k not in old: continue
        (ot, op, of), (nt, np_, nf) = old[k], new[k]
        rt.append(nt / ot); rp.append(np_ / op); rf.append(nf / of)
        print(f"| {k[0]} | {k[1]} | {ot:.1f} -> {nt:.1f} | {100*(nt/ot-1):+.0f}% | {op:.0f} -> {np_:.0f} | {100*(np_/op-1):+.0f}% "
              f"| {of:.0f} -> {nf:.0f} | {100*(nf/of-1):+.0f}% |")
    print(f"\ngeomean over {len(rt)} points: TTT/GPU {100*(gmean(rt)-1):+.1f}%, P90 {100*(gmean(rp)-1):+.1f}%, TTFT {100*(gmean(rf)-1):+.1f}%")
