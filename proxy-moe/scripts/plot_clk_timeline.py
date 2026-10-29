# Effective GFXCLK per partition over time, from mb_skew.py --dump (DPM run) calibrated by a locked run:
#   eff_MHz(rank, iter) = LOCK_MHz * locked_p50_us(rank) / gemm_us(rank, iter)   (compute-bound GEMM, time ~ 1/clock)
# sysfs freq1_input is a smoothed value (+-10 MHz while the GEMM time swings 2x), so it cannot show this.
# Usage: plot_clk_timeline.py <mb.json> <lock_mhz> <g0_p50,...,g7_p50 us> <out.png> [window_s=1.0] [zoom_ms=100]
import json, sys, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
mbj, lock, base, out = sys.argv[1], float(sys.argv[2]), [float(x) for x in sys.argv[3].split(",")], sys.argv[4]
win = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0; zoom = float(sys.argv[6]) if len(sys.argv) > 6 else 100.0
m = json.load(open(mbj)); gpus = m["gpus"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HUE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # socket 0..3, reference categorical order
t_mid = st.median(m["start"][0]); a = t_mid - win / 2
fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor=SURF)
for ax, (lo, hi), lw, ms in ((axes[0], (0, win * 1e3), 1.1, 0), (axes[1], (win * 500 - zoom / 2, win * 500 + zoom / 2), 1.6, 3.5)):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c9c8c3")
    for k, g in enumerate(gpus):
        pts = [((t - a) * 1e3, lock * base[g] / d) for t, d in zip(m["start"][k], m["compute_us"][k]) if lo <= (t - a) * 1e3 <= hi]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=HUE[g // 2], lw=lw, ls="-" if g % 2 == 0 else (0, (3, 2)),
                marker=("o" if g % 2 == 0 else "s") if ms else None, ms=ms, label=f"GPU {g}")
    ax.set_xlim(lo, hi); ax.set_ylabel("effective GFXCLK (MHz)", color=INK2)
axes[0].axvspan(win * 500 - zoom / 2, win * 500 + zoom / 2, color="#d9d8d3", alpha=0.5, lw=0)
axes[0].set_title(f"Effective GFXCLK per partition under DPM, 8 ranks, identical GEMM + 2 MiB all-reduce per iteration ({win:.0f} s)",
                  color=INK, fontsize=11, loc="left")
axes[1].set_title(f"Zoom: shaded {zoom:.0f} ms (one marker = one iteration; solid = partition 0, dashed = partition 1 of a socket)",
                  color=INK, fontsize=10, loc="left")
axes[1].set_xlabel("time (ms)", color=INK2)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=8, fontsize=9, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, 0.03))
fig.text(0.5, 0.008, f"effective MHz = {lock:.0f} x (GEMM time when all sockets were locked at {lock:.0f} MHz) / (GEMM time this iteration)",
         color=INK2, fontsize=8, ha="center")
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out, dpi=130, facecolor=SURF); print("wrote", out)
