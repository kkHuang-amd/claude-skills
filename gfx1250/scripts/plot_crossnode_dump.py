"""Plot the cross-node per-layer residual-stream comparison (E28).

gfx950 (TP2, a4w4 MoE, bf16 absorb via ablation) vs gfx1250 (TP1, a8w4 MoE, bf16 absorb).
Compares `input`, `post_attn` (attention-sublayer output, the clean non-MoE signal),
and `post_layer` (post-MoE, the a4w4-vs-a8w4 control). Both dumps now use the
post-attention-split hook (gfx1250 re-dumped 2026-07-09, no longer stale).

Usage:
  python3 scripts/plot_crossnode_dump.py \
      --g950 hs_dump_gfx950.json --g1250 hs_dump_gfx1250.json --out crossnode_dump_compare.png
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rel_l2(a, b):
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(y * y for y in b))
    return num / den if den > 0 else float("nan")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--g950", default=os.path.join(here, "hs_dump_gfx950.json"))
    ap.add_argument("--g1250", default=os.path.join(here, "hs_dump_gfx1250.json"))
    ap.add_argument("--out", default=os.path.join(here, "crossnode_dump_compare.png"))
    ap.add_argument("--dense", type=int, default=3, help="first_k_dense_replace (dense layers 0..dense-1)")
    args = ap.parse_args()

    g50 = json.load(open(args.g950))["layers"]
    g12 = json.load(open(args.g1250))["layers"]
    n = min(len(g50), len(g12))
    layers = list(range(n))

    in_rel, pa_rel, pl_rel = [], [], []
    n50_in, n12_in, n50_pl, n12_pl = [], [], [], []
    for i in range(n):
        a, b = g50[i], g12[i]
        ia, ib = a["input"], b["input"]
        pa, pb = a["post_layer"], b["post_layer"]
        aa, ab = a.get("post_attn"), b.get("post_attn")
        in_rel.append(rel_l2(ia["slice64"], ib["slice64"]) if ia and ib else float("nan"))
        pa_rel.append(rel_l2(aa["slice64"], ab["slice64"]) if aa and ab else float("nan"))
        pl_rel.append(rel_l2(pa["slice64"], pb["slice64"]))
        n50_in.append(ia["norm"] if ia else float("nan"))
        n12_in.append(ib["norm"] if ib else float("nan"))
        n50_pl.append(pa["norm"])
        n12_pl.append(pb["norm"])

    dense = args.dense
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # ---- Panel 1: rel_l2 (log) ----
    ax1.axvspan(-0.5, dense - 0.5, color="tab:green", alpha=0.10, label=f"dense layers 0-{dense-1} (no MoE)")
    ax1.axvspan(dense - 0.5, n - 0.5, color="tab:red", alpha=0.06, label=f"MoE layers {dense}-{n-1}")
    ax1.axhline(1e-2, color="gray", ls="--", lw=1, label="TP/bf16 floor ~1e-2")
    ax1.plot(layers, in_rel, "o-", ms=4, color="tab:blue", label="input rel_l2")
    ax1.plot(layers, pa_rel, "D-", ms=4, color="tab:purple", label="post_attn rel_l2 (non-MoE signal)")
    ax1.plot(layers, pl_rel, "s-", ms=4, color="tab:orange", label="post_layer rel_l2")
    ax1.axvline(dense, color="tab:red", ls=":", lw=1.5)
    ax1.annotate(
        f"L{dense}: first MoE layer\ninput still matched ({in_rel[dense]:.3f}),\npost_layer jumps to {pl_rel[dense]:.2f}",
        xy=(dense, pl_rel[dense]), xytext=(dense + 6, pl_rel[dense] * 0.9),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="tab:red"),
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("rel_l2 of slice64 (last token, first 64 dims)")
    ax1.set_title(
        "Cross-node per-layer residual diff: gfx950 (a4w4 MoE) vs gfx1250 (a8w4 MoE)\n"
        "both bf16 absorb (gfx950 via SGLANG_DISABLE_QUARK_ABSORB_FP4 ablation); "
        "non-MoE matches to floor, divergence starts exactly at the first MoE layer",
        fontsize=10,
    )
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # ---- Panel 2: norms ----
    ax2.axvspan(-0.5, dense - 0.5, color="tab:green", alpha=0.10)
    ax2.axvspan(dense - 0.5, n - 0.5, color="tab:red", alpha=0.06)
    ax2.plot(layers, n50_pl, "o-", ms=3, color="tab:orange", label="post_layer norm  gfx950 (a4w4)")
    ax2.plot(layers, n12_pl, "o--", ms=3, color="tab:red", label="post_layer norm  gfx1250 (a8w4)")
    ax2.plot(layers, n50_in, "^-", ms=3, color="tab:blue", label="input norm  gfx950 (a4w4)")
    ax2.plot(layers, n12_in, "^--", ms=3, color="tab:cyan", label="input norm  gfx1250 (a8w4)")
    ax2.axvline(dense, color="tab:red", ls=":", lw=1.5)
    ax2.set_ylabel("residual-stream norm (last token)")
    ax2.set_xlabel("decoder layer")
    ax2.set_title("Residual-stream magnitude: gfx950 (a4w4) grows LARGER at depth despite higher accuracy", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    # also print a compact table
    print(f"{'L':>3} {'zone':<5} {'in_relL2':>9} {'pa_relL2':>9} {'pl_relL2':>9} {'n50_pl':>8} {'n12_pl':>8}")
    for i in range(n):
        z = "dense" if i < dense else "moe"
        print(f"{i:>3} {z:<5} {in_rel[i]:9.4f} {pa_rel[i]:9.4f} {pl_rel[i]:9.4f} {n50_pl[i]:8.2f} {n12_pl[i]:8.2f}")


if __name__ == "__main__":
    main()
