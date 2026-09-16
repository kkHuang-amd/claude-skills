#!/usr/bin/env python3
"""Cross-rank view of one captured window. This is the tool that attributes.

    python3 trace_ranks.py <trace dir> [<trace dir> ...]

In a DP-attention deployment every rank's step ends together, so a single
rank's trace tells you nothing about who did the work: the fused MoE
all-to-all kernel absorbs the wait, and a lightly loaded rank shows a huge
`moe`/`comm` time while doing almost nothing. Comparing ranks separates them:

  compute  = attn + gemm + quant + norm_rope + sample   (tracks own tokens)
  barrier  = moe + comm                                 (runs OPPOSITE to own tokens)

Read `barrier` as "waiting for the group", not as MoE cost, whenever it
anti-correlates with token count. The straggler — the rank with the most
compute — is what sets the step time for all eight.
"""
import collections
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import (attribute, classify, load, rank_of,  # noqa: E402
                          verify_classes)

COMPUTE = ("attn", "gemm", "quant", "norm_rope", "sample")
BARRIER = ("moe", "comm")


def _overlap(lo, hi, lo2, hi2):
    return lo < hi2 and lo2 < hi


def rows_for(path):
    steps, kernels = load(path)
    owned = attribute(steps, kernels)
    # Split DSPARK draft vs full-model verify; they share a bs and must not
    # share a row (see verify_classes). Same grouping as trace_summary.py.
    vclass = verify_classes(steps, owned)
    per_type = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, s in enumerate(steps):
        roles = collections.defaultdict(float)
        for dur, name in owned.get(i, []):
            roles[classify(name)] += dur
        name = s["type"] + vclass.get(i, "")
        # Keyed by (type[+class], bs): different bs = different work.
        key = f"{name} bs={s['bs']}" if s["bs"] is not None else name
        rec = per_type[key]
        rec["step_ms"].append(s["ms"])
        rec["toks"].append(s["toks"] or 0)
        rec["compute"].append(sum(roles[r] for r in COMPUTE) / 1000.0)
        rec["barrier"].append(sum(roles[r] for r in BARRIER) / 1000.0)
        rec["attn"].append(roles["attn"] / 1000.0)
        rec["gemm"].append(roles["gemm"] / 1000.0)
        rec["_lo"].append(s["lo"])
        rec["_hi"].append(s["hi"])
        rec["_type"].append(s["type"])
        rec["_class"].append(vclass.get(i, ""))
    return per_type, steps


def report(d):
    files = sorted(Path(d).glob("*.trace.json.gz"), key=rank_of)
    if not files:
        print(f"\n=== {d}: no trace files")
        return
    print(f"\n=== {d}  ({len(files)} ranks)")
    table = {}
    all_steps = {}
    for f in files:
        rows, steps = rows_for(f)
        table[rank_of(f)] = rows
        all_steps[rank_of(f)] = steps

    stypes = sorted({t for r in table.values() for t in r},
                    key=lambda t: -sum(len(r[t]["step_ms"]) for r in table.values() if t in r))
    for stype in stypes:
        print(f"\n  -- {stype}")
        # Totals and means, never medians. A step type can be thousands of
        # highly skewed entries (TARGET_VERIFY: p50 0.4 ms, mean 2.9 ms), where
        # most steps carry almost no kernel time and a few carry it all -- the
        # median then reads 0.0 and hides the entire workload.
        print(f"  {'rank':>4s} {'steps':>6s} {'toks/step':>9s} {'tot ms':>8s}"
              f" {'ms/step':>8s} {'compute':>8s} {'barrier':>8s} {'attn':>7s} {'gemm':>7s}")
        summary = []
        for rank in sorted(table):
            r = table[rank].get(stype)
            if not r:
                continue
            n = len(r["step_ms"])
            row = (rank, n, st.fmean(r["toks"]), sum(r["step_ms"]),
                   st.fmean(r["step_ms"]), st.fmean(r["compute"]), st.fmean(r["barrier"]),
                   st.fmean(r["attn"]), st.fmean(r["gemm"]))
            summary.append(row)
            print(f"  {row[0]:4d} {row[1]:6d} {row[2]:9.0f} {row[3]:8.0f}"
                  f" {row[4]:8.2f} {row[5]:8.2f} {row[6]:8.2f} {row[7]:7.2f} {row[8]:7.2f}")
        if len(summary) > 2:
            toks = [r[2] for r in summary]
            comp = [r[5] for r in summary]
            barr = [r[6] for r in summary]
            step = [r[4] for r in summary]
            lead = max(summary, key=lambda r: r[5])
            # Spearman-free check: does barrier fall as tokens rise?
            # Only claim a correlation when the token counts actually vary.
            # TARGET_VERIFY steps carry no token annotation, so every rank reads
            # 0 and any "anti-correlated" verdict would be pure noise stated
            # with confidence -- the worst kind of output.
            order = sorted(range(len(summary)), key=lambda i: toks[i])
            if max(toks) - min(toks) < 1:
                trend = "no token spread across ranks -- correlation not evaluated"
            else:
                trend = ("anti-correlated with tokens (=> it is a WAIT)"
                         if barr[order[0]] > barr[order[-1]] else "tracks tokens")
            print(f"\n     step ms spread {min(step):.0f}-{max(step):.0f}"
                  f" (max/min {max(step) / max(min(step), 1e-9):.2f}x)"
                  f"   token spread {min(toks):.0f}-{max(toks):.0f}"
                  f" ({max(toks) / max(min(toks), 1):.1f}x)")
            print(f"     compute spread {min(comp):.1f}-{max(comp):.1f} ms"
                  f"   barrier {min(barr):.1f}-{max(barr):.1f} ms, {trend}")
            print(f"     straggler: rank {lead[0]} with {lead[5]:.2f} ms compute/step"
                  f" at {lead[2]:.0f} tokens -- it sets the step for every rank")

    # Full-model verify wall with no concurrent EXTEND anywhere in the group.
    # Overlap is by GPU-annotation timestamps (this capture's clocks align).
    extends = [(r, s["lo"], s["hi"])
               for r, ss in all_steps.items() for s in ss if s["type"] == "EXTEND"]
    print("\n  -- TARGET_VERIFY full, no concurrent EXTEND in the 8-rank group")
    ext_ranks = sorted({r for r, _, _ in extends})
    print(f"     EXTEND intervals: {len(extends)} (ranks {ext_ranks})")
    print(f"  {'rank':>4s} {'n_full':>6s} {'n_free':>6s} {'n_hit':>6s}"
          f" {'p50':>8s} {'mean':>8s} {'min':>8s} {'max':>8s}")
    rank_p50 = []
    for rank in sorted(table):
        free, hit = [], []
        for stype, rec in table[rank].items():
            if rec["_type"][0] != "TARGET_VERIFY" or rec["_class"][0] != " full":
                continue
            for ms, lo, hi in zip(rec["step_ms"], rec["_lo"], rec["_hi"]):
                if any(_overlap(lo, hi, elo, ehi) for _, elo, ehi in extends):
                    hit.append(ms)
                else:
                    free.append(ms)
        n_full = len(free) + len(hit)
        if n_full == 0:
            continue
        if free:
            p50 = st.median(free)
            rank_p50.append((rank, p50, len(free), min(free), max(free)))
            print(f"  {rank:4d} {n_full:6d} {len(free):6d} {len(hit):6d}"
                  f" {p50:8.1f} {st.fmean(free):8.1f} {min(free):8.1f} {max(free):8.1f}")
        else:
            print(f"  {rank:4d} {n_full:6d} {len(free):6d} {len(hit):6d}"
                  f" {'-':>8s} {'-':>8s} {'-':>8s} {'-':>8s}")
    if len(rank_p50) >= 3:
        p50s = [p for _, p, _, _, _ in rank_p50]
        print(f"\n     free-of-EXTEND full-verify step wall: "
              f"{len(rank_p50)} ranks, p50 spread {min(p50s):.1f}-{max(p50s):.1f} ms"
              f" (max/min {max(p50s) / max(min(p50s), 1e-9):.3f}x)")
    elif rank_p50:
        print("\n     fewer than 3 ranks with EXTEND-free full verify -- not enough")
    else:
        print("\n     no EXTEND-free full TARGET_VERIFY steps in this window")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        report(d)
