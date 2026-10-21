#!/usr/bin/env python3
"""TTT/GPU vs P90 interactivity, GB300 vs MI355X, from the two RESULTS.md files.
Usage: python3 plot_ttt_vs_p90.py [out.png]"""
import os, re, sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {
    "MI355X": os.path.join(HERE, "..", "fixed_seq_len", "RESULTS.md"),
    "GB300": os.path.join(HERE, "RESULTS.md"),
}


def parse(path):
    out, cur = {}, None
    for l in open(path):
        m = re.match(r"# (\d+)/(\d+)\s+(\S+)", l)
        if m:
            cur = (f"{m[1]}/{m[2]}", m[3])
            continue
        c = [x.strip() for x in l.split("|")[1:-1]]
        if cur and len(c) == 11 and c[0].isdigit():
            out.setdefault(cur, []).append((int(c[3]), float(c[5]), float(c[10])))
        elif not l.startswith("|"):
            cur = None if l.strip() else cur
    return out


SERIES = [  # (platform, recipe, label, style)
    ("GB300", "tp4ep1", "GB300 tp4ep1 (12L, 4 GPU)", dict(color="tab:green", marker="o", ls="-")),
    ("MI355X", "tp4ep1", "MI355X tp4ep1 (12L, 4 GPU)", dict(color="tab:red", marker="s", ls="-")),
    ("MI355X", "tp8ep1", "MI355X tp8ep1 (24L, 8 GPU)", dict(color="tab:red", marker="^", ls="--", alpha=0.5)),
]
data = {p: parse(f) for p, f in SRC.items()}
shapes = ["1024/1024", "8192/1024", "70000/300"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, shape in zip(axes, shapes):
    for plat, recipe, label, st in SERIES:
        pts = sorted(data[plat].get((shape, recipe), []))
        if not pts:
            continue
        x = [p[2] for p in pts]
        y = [p[1] for p in pts]
        ax.plot(x, y, label=label, **st)
        if recipe == "tp4ep1":
            for conc, yy, xx in pts:
                ax.annotate(str(conc), (xx, yy), textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=st["color"])
    ax.set_title(f"ISL/OSL {shape}")
    ax.set_xlabel("P90 interactivity (tok/s/user) = 1000 / p90 ITL")
    ax.set_ylabel("TTT/GPU (tok/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Proxy MoE fixed_seq_len: TTT/GPU vs P90 interactivity (labels = concurrency)")
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ttt_vs_p90.png")
fig.savefig(out, dpi=130)
print(out)
