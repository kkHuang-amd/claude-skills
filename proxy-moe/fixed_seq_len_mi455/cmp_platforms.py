#!/usr/bin/env python3
"""MI455 TP8 (EP1 / EP8) vs MI355X TP4 EP1 vs GB300 TP4 EP1, all 12 layers.

Parses the RESULTS.md tables of the three sweeps and prints markdown:
TTT/GPU, P90 interactivity and median TTFT per (shape, CONC), plus geomean ratios.
Usage: python3 cmp_platforms.py > /tmp/cmp.md
"""
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MI455 = os.path.join(HERE, "RESULTS.md")
MI355 = os.path.join(HERE, "..", "fixed_seq_len", "RESULTS.md")
GB300 = os.path.join(HERE, "..", "fixed_seq_len_gb300", "RESULTS.md")
SHAPES = ["1024/1024", "8192/1024", "70000/300"]


def num(s):
    return float(s.strip().strip("*"))


def parse_report(path, recipe="tp4ep1"):
    """report.py tables: '# ISL/OSL  recipe' then rows; -> {(shape, conc): (ttt_gpu, p90, ttft)}"""
    out, cur = {}, None
    for line in open(path):
        m = re.match(r"^# (\d+/\d+)\s+(\S+)", line)
        if m:
            cur = m.group(1) if m.group(2) == recipe else None
            continue
        if line.startswith("## "):
            cur = None
        if cur and re.match(r"^\| \d", line):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) == 11:
                out[(cur, int(c[3]))] = (num(c[5]), num(c[10]), num(c[7]))
    return out


def parse_mi455(path):
    """Summary tables: '### ISL/OSL' then CONC | EP1 TTT | EP8 TTT | EP1 P90 | EP8 P90 | EP1 TTFT | EP8 TTFT"""
    ep1, ep8, cur = {}, {}, None
    for line in open(path):
        m = re.match(r"^### (\d+/\d+)\s*$", line)
        if m:
            cur = m.group(1)
            continue
        if line.startswith("## "):
            cur = None
        if cur and re.match(r"^\| \d", line):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            k = (cur, int(c[0]))
            ep1[k] = (num(c[1]), num(c[3]), num(c[5]))
            ep8[k] = (num(c[2]), num(c[4]), num(c[6]))
    return ep1, ep8


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ep1, ep8 = parse_mi455(MI455)
    mi355, gb = parse_report(MI355), parse_report(GB300)
    # Best MI455 recipe per point, by TTT/GPU.
    best = {k: (ep1[k], "EP1") if ep1[k][0] >= ep8[k][0] else (ep8[k], "EP8") for k in ep1}
    ratios = {s: {"355_ttt": [], "gb_ttt": [], "355_p90": [], "gb_p90": []} for s in SHAPES}
    for s in SHAPES:
        concs = sorted(c for (sh, c) in best if sh == s and (sh, c) in mi355 and (sh, c) in gb)
        print(f"### {s}\n")
        print("| CONC | MI455 EP1 TTT/GPU | MI455 EP8 TTT/GPU | MI355X TTT/GPU | GB300 TTT/GPU "
              "| MI455 best / MI355X | MI455 best / GB300 "
              "| MI455 P90 intvty (best-TTT recipe) | MI355X P90 | GB300 P90 "
              "| MI455 TTFT ms (best-TTT recipe) | MI355X TTFT | GB300 TTFT |")
        print("|---" * 13 + "|")
        for c in concs:
            k = (s, c)
            (bt, bp, bf), tag = best[k]
            m, g = mi355[k], gb[k]
            r355, rgb = bt / m[0], bt / g[0]
            ratios[s]["355_ttt"].append(r355)
            ratios[s]["gb_ttt"].append(rgb)
            ratios[s]["355_p90"].append(bp / m[1])
            ratios[s]["gb_p90"].append(bp / g[1])
            print(f"| {c} | {ep1[k][0]:.1f} | {ep8[k][0]:.1f} | {m[0]:.1f} | {g[0]:.1f} "
                  f"| {r355:.2f}x | {rgb:.2f}x "
                  f"| {bp:.1f} ({tag}) | {m[1]:.1f} | {g[1]:.1f} "
                  f"| {bf:.0f} | {m[2]:.0f} | {g[2]:.0f} |")
        print()
    print("### Geomean MI455 (best of EP1/EP8) ratio, over the common CONC points\n")
    print("| Shape | TTT/GPU vs MI355X | TTT/GPU vs GB300 | P90 intvty vs MI355X | P90 intvty vs GB300 |")
    print("|---|---|---|---|---|")
    allr = {k: [] for k in ratios[SHAPES[0]]}
    for s in SHAPES:
        r = ratios[s]
        for k in allr:
            allr[k] += r[k]
        print(f"| {s} | {gmean(r['355_ttt']):.2f}x | {gmean(r['gb_ttt']):.2f}x "
              f"| {gmean(r['355_p90']):.2f}x | {gmean(r['gb_p90']):.2f}x |")
    print(f"| all | {gmean(allr['355_ttt']):.2f}x | {gmean(allr['gb_ttt']):.2f}x "
          f"| {gmean(allr['355_p90']):.2f}x | {gmean(allr['gb_p90']):.2f}x |")


if __name__ == "__main__":
    main()
