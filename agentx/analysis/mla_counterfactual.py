#!/usr/bin/env python3
"""What does the step wall become if the MLA decode kernel gets faster?

    python3 mla_counterfactual.py <trace dir> [--kernel PAT[,PAT...]]

Answers, from an EXISTING trace and with no GPU, the question that otherwise
needs a fake-kernel run: MLA is 7.47 ms/step of direct cost, but the ranks wait
for each other inside `megamoe_prepare_compact` (a spin-wait, r = -0.942), so a
faster MLA should pay twice -- once directly, once by shortening the wait.
"Should" is a hypothesis; this file puts a number on it.

METHOD, and its assumptions, which are the whole argument:

  A decode step is cross-rank synchronised (per-rank walls agree to 1.002x),
  so model the group wall as

      W = max_r(O_r) + floor          O_r = rank r's own work, wait excluded
                                      floor = protocol/tail not attributable
                                              to any rank's own work

  `floor` is not assumed: it is MEASURED per step group as W - max_r(O_r), and
  then held constant under the counterfactual. Scaling MLA by k gives

      W(k) = max_r(O_r + (k-1) * M_r) + floor

  k=0 is the fake-kernel arm, k=2,3 the accuracy-preserving duplication arm.

  The saving at k=0 is max_r(O_r) - max_r(O_r - M_r). Divide it by mean_r(M_r),
  which is what a per-rank kernel table credits MLA with, and the ratio is the
  MULTIPLIER. >1 means the wait really is downstream of MLA; ~1 means MLA is
  worth only its own time and the imbalance is driven by something else.

WHAT THIS IS NOT. It is a first-order critical-path model, not a simulation.
Three things it cannot know, all of which bias it OPTIMISTIC, so treat its
number as an upper bound to be validated on the GPU:
  - slack is assumed fungible: removing work from the slowest rank is assumed
    to shorten the barrier rather than expose a different serialisation;
  - `floor` is held constant, though part of it is the prepare protocol which
    may not shrink;
  - a rank's MLA is removed in place, ignoring any cache/occupancy interaction
    with what follows it.

`O_r` is a UNION of kernel intervals (elapsed), never a sum of durations -- see
FINDINGS' method rules. MI355X is serial so the two nearly agree; the ratio is
printed so a platform where they do not is visible immediately.
"""
import collections
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import load, rank_of, verify_classes  # noqa: E402

# Default target: both MLA decode variants plus the split-K reduction that only
# exists because of the split one. Kept as one knob because a fake/faster MLA
# would replace the whole group; `--kernel` overrides it to probe a subset.
MLA_PAT = "_paged_decode_split_kernel,_paged_decode_fused_kernel,_paged_decode_reduce_kernel"
# The cross-rank spin-wait. Excluded from `own work` -- that is the point.
WAIT_PAT = "megamoe_prepare_compact"

SCALES = (0.0, 0.25, 0.5, 2.0, 3.0)


def union_ms(iv):
    """Elapsed span covered by (lo, hi) intervals, in ms. Never a sum."""
    if not iv:
        return 0.0
    iv = sorted(iv)
    tot, clo, chi = 0.0, *iv[0]
    for lo, hi in iv[1:]:
        if lo > chi:
            tot += chi - clo
            clo, chi = lo, hi
        else:
            chi = max(chi, hi)
    return (tot + chi - clo) / 1000.0


def load_rank(path, mla_pats):
    """-> list of per-full-verify-step dicts for one rank."""
    steps, kernels = load(path)
    # Re-attribute keeping ts, which trace_common.attribute() drops.
    owned = collections.defaultdict(list)
    for ts, dur, name in kernels:
        for i, s in enumerate(steps):
            if s["lo"] <= ts <= s["hi"]:
                owned[i].append((ts, dur, name))
                break
    vc = verify_classes(steps, {i: [(d, n) for _t, d, n in v]
                               for i, v in owned.items()})
    out = []
    for i, s in enumerate(steps):
        if s["type"] != "TARGET_VERIFY" or vc.get(i) != " full":
            continue
        ks = owned.get(i, [])
        own, mla, wait, osum = [], [], [], 0.0
        for ts, dur, name in ks:
            if WAIT_PAT in name:
                wait.append((ts, ts + dur))
            else:
                # `sum` must cover exactly the same kernels as `own`, or the
                # concurrency check reports the excluded wait as overlap.
                osum += dur
                own.append((ts, ts + dur))
                if any(p in name for p in mla_pats):
                    mla.append((ts, ts + dur))
        out.append({
            "rank": rank_of(path), "bs": s["bs"], "lo": s["lo"], "hi": s["hi"],
            "wall": s["ms"], "own": union_ms(own), "mla": union_ms(mla),
            "wait": union_ms(wait), "sum": osum / 1000.0,
            "nmla": len(mla),
        })
    return out


def group_steps(all_steps):
    """Cluster per-rank steps into cross-rank decode iterations by overlap."""
    groups, cur = [], []
    for s in sorted(all_steps, key=lambda x: x["lo"]):
        if cur and s["lo"] < min(c["hi"] for c in cur) \
                and s["rank"] not in {c["rank"] for c in cur}:
            cur.append(s)
        else:
            if cur:
                groups.append(cur)
            cur = [s]
    if cur:
        groups.append(cur)
    return groups


def main(d, mla_pats):
    all_steps = []
    for f in sorted(Path(d).glob("*.trace.json.gz"), key=rank_of):
        all_steps += load_rank(f, mla_pats)
    if not all_steps:
        print("no full TARGET_VERIFY steps found")
        return

    conc = st.fmean(s["sum"] / s["own"] for s in all_steps if s["own"] > 0)
    print(f"=== {d}")
    print(f"target kernels: {','.join(mla_pats)}")
    print(f"{len(all_steps)} full TARGET_VERIFY steps over "
          f"{len({s['rank'] for s in all_steps})} ranks; "
          f"sum/union = {conc:.3f} (1.0 = serial, so union==sum)\n")

    groups = [g for g in group_steps(all_steps) if len(g) >= 2]
    if not groups:
        print("no cross-rank step groups -- ranks did not overlap in time")
        return

    print(f"{len(groups)} cross-rank step groups "
          f"(ranks/group: {min(len(g) for g in groups)}-{max(len(g) for g in groups)})\n")

    print("per rank, over its own full-verify steps (median ms/step).")
    print("us/call averages split+fused+reduce together, so it is lower than")
    print("the per-variant numbers in mi355x-decode-trace.md -- not comparable.")
    print(f"{'rank':>4s} {'bs':>3s} {'n':>3s} {'wall':>7s} {'own':>7s} "
          f"{'mla':>7s} {'wait':>7s} {'us/call':>8s}")
    by_rank = collections.defaultdict(list)
    for s in all_steps:
        by_rank[s["rank"]].append(s)
    for r in sorted(by_rank):
        v = by_rank[r]
        pc = [s["mla"] * 1000 / s["nmla"] for s in v if s["nmla"]]
        print(f"{r:4d} {st.median([s['bs'] for s in v]):3.0f} {len(v):3d} "
              f"{st.median([s['wall'] for s in v]):7.2f} "
              f"{st.median([s['own'] for s in v]):7.2f} "
              f"{st.median([s['mla'] for s in v]):7.2f} "
              f"{st.median([s['wait'] for s in v]):7.2f} "
              f"{st.median(pc) if pc else 0:8.1f}")

    # --- the counterfactual, per group ---
    rows = []
    for g in groups:
        W = st.median([s["wall"] for s in g])
        O = {s["rank"]: s["own"] for s in g}
        M = {s["rank"]: s["mla"] for s in g}
        omax = max(O.values())
        floor = W - omax
        slowest = max(O, key=O.get)
        row = {"W": W, "floor": floor, "mean_mla": st.fmean(M.values()),
               "mla_of_slowest": M[slowest], "nranks": len(g),
               "walls": max(s["wall"] for s in g) / min(s["wall"] for s in g)}
        for k in SCALES:
            row[k] = max(O[r] + (k - 1) * M[r] for r in O) + floor
        # Residual imbalance, which is the directly testable prediction: with
        # MLA at scale k, how much slack is each rank left holding at the
        # barrier? `crit` names the rank the model puts on the critical path,
        # because the multiplier is only stable while that rank does not change.
        row["bal"] = {}
        for k in (0.0, 1.0):
            adj = {r: O[r] + (k - 1) * M[r] for r in O}
            top = max(adj.values())
            row["bal"][k] = ({r: top - adj[r] for r in adj},
                             max(adj, key=adj.get), top - min(adj.values()))
        rows.append(row)

    W = st.median([r["W"] for r in rows])
    floor = st.median([r["floor"] for r in rows])
    mean_mla = st.fmean([r["mean_mla"] for r in rows])
    print(f"\nmodel fit, median over {len(rows)} groups:")
    print(f"  observed step wall            {W:7.2f} ms")
    print(f"  max_r(own work)               {W - floor:7.2f} ms")
    print(f"  implied floor (wall - max)    {floor:7.2f} ms  "
          f"({100 * floor / W:.0f} % of the wall)")
    print(f"  per-rank wall agreement       {st.median([r['walls'] for r in rows]):7.3f}x  "
          f"(model assumes ~1.00)")
    print(f"  mean_r(MLA) = what a kernel table credits MLA with "
          f"{mean_mla:.2f} ms")
    print(f"  MLA of the rank ON the critical path              "
          f"{st.fmean([r['mla_of_slowest'] for r in rows]):.2f} ms")

    print(f"\nPREDICTION -- step wall vs MLA cost scale k "
          f"(k=0 is the fake kernel, k=2/3 the duplication arm):")
    print(f"{'k':>5s} {'wall ms':>9s} {'delta':>8s} {'vs k=1':>8s} "
          f"{'multiplier':>11s}")
    for k in sorted(set(SCALES) | {1.0}):
        w = W if k == 1.0 else st.median([r[k] for r in rows])
        d = w - W
        naive = (k - 1) * mean_mla
        mult = d / naive if abs(naive) > 1e-9 else float("nan")
        print(f"{k:5.2f} {w:9.2f} {d:+8.2f} {naive:+8.2f} {mult:11.2f}")
    print("\nRESIDUAL IMBALANCE -- the directly testable half of the prediction.")
    print("Slack each rank is left holding at the barrier, ms:")
    print(f"{'rank':>4s} {'k=1 model':>10s} {'k=1 observed':>13s} {'k=0 model':>10s}")
    for r in sorted(by_rank):
        m1 = st.median([g["bal"][1.0][0][r] for g in rows if r in g["bal"][1.0][0]])
        m0 = st.median([g["bal"][0.0][0][r] for g in rows if r in g["bal"][0.0][0]])
        obs = st.median([s["wait"] for s in by_rank[r]])
        print(f"{r:4d} {m1:10.2f} {obs:13.2f} {m0:10.2f}")
    sp1 = st.median([g["bal"][1.0][2] for g in rows])
    sp0 = st.median([g["bal"][0.0][2] for g in rows])
    c1 = collections.Counter(g["bal"][1.0][1] for g in rows).most_common(1)[0]
    c0 = collections.Counter(g["bal"][0.0][1] for g in rows).most_common(1)[0]
    print(f"\ncross-rank spread of own work:  k=1 {sp1:.2f} ms -> k=0 {sp0:.2f} ms")
    print(f"critical rank: k=1 rank {c1[0]} in {c1[1]}/{len(rows)} groups, "
          f"k=0 rank {c0[0]} in {c0[1]}/{len(rows)} groups")
    print("The k=1 model column is a CHECK, not a result: it should track the")
    print("observed prepare wait. Where it does, the barrier model holds.")
    print("If the critical rank changes between k=1 and k=0 the multiplier is")
    print("not constant, and only the k it was measured at can be quoted.")

    # Direction A (`--load-balance-method total_tokens`) and a faster MLA are
    # NOT additive: both are paid out of the same slack. Price them jointly
    # here, offline, before either costs an hour of GPU. "Balanced" is the
    # upper bound for A -- perfectly equal own work, which is more than a KV
    # token balancer can deliver.
    # The A/B falsified `total_tokens` as a way to equalise ranks, but it did not
    # touch the model: the step wall is still max_r(own) + floor. So ask the
    # question the kernel can actually act on -- what if MLA's DISPERSION across
    # ranks were removed, every rank paying what the cheapest rank pays today?
    # That is what a straggler-aware kernel buys, and unlike a load balancer it
    # does not require moving any work between ranks.
    print("\nMLA DISPERSION -- what a straggler-aware kernel could buy:")
    print(f"{'scenario':<40s} {'wall ms':>8s} {'saving':>8s}")
    base_rows = []
    for g, grp in zip(rows, groups):
        O = {s["rank"]: s["own"] for s in grp}
        M = {s["rank"]: s["mla"] for s in grp}
        base_rows.append((g, O, M))
    for label, mode in (("as measured", None),
                        ("MLA equalised to the cheapest rank", "min"),
                        ("MLA equalised to the mean", "mean"),
                        ("MLA free (floor of this family)", "zero")):
        vals = []
        for g, O, M in base_rows:
            tgt = (0.0 if mode == "zero" else
                   min(M.values()) if mode == "min" else
                   st.fmean(M.values()) if mode == "mean" else None)
            adj = {r: (O[r] if tgt is None else O[r] - M[r] + tgt) for r in O}
            vals.append(max(adj.values()) + g["floor"])
        w = st.median(vals)
        print(f"{label:<40s} {w:8.2f} {w - W:+8.2f}")
    print("Equalising costs nothing in total work -- it only moves the critical")
    print("rank's MLA down to what another rank already achieves today, so it is")
    print("reachable by the kernel alone. Contrast with the balancer rows below,")
    print("whose route the total_tokens A/B falsified.")

    print("\nDIRECTION A vs MLA -- same slack, so they do not add up:")
    print(f"{'scenario':<34s} {'wall ms':>8s} {'saving':>8s}")
    kb = 1.0 - 7.47 / mean_mla       # MLA at B200 speed: the published gap
    for label, k, bal in (("as measured", 1.0, False),
                          ("balanced ranks (A, upper bound)", 1.0, True),
                          ("MLA at B200 speed", kb, False),
                          ("both", kb, True),
                          ("MLA free (fake kernel)", 0.0, False),
                          ("MLA free + balanced", 0.0, True)):
        vals = []
        for g, grp in zip(rows, groups):
            O = {s["rank"]: s["own"] for s in grp}
            M = {s["rank"]: s["mla"] for s in grp}
            adj = {r: O[r] + (k - 1) * M[r] for r in O}
            top = st.fmean(adj.values()) if bal else max(adj.values())
            vals.append(top + g["floor"])
        w = st.median(vals)
        print(f"{label:<34s} {w:8.2f} {w - W:+8.2f}")
    print("Read the last rows against the component table: MLA is credited")
    print("+7.47 ms and the cross-rank idle +9.96, but they are one number.")

    print("\n'vs k=1' is the naive prediction: MLA credited at its per-rank mean,")
    print("no wait effect. multiplier = actual / naive. >1 means the wait is")
    print("downstream of MLA and a speed-up pays twice; ~1 means it does not.")
    print("\nUpper bound, not a forecast -- the three optimistic assumptions are")
    print("in this file's docstring. k=2 is the accuracy-exact arm that")
    print("validates it on the GPU without touching accept len or OSL.")


if __name__ == "__main__":
    args = sys.argv[1:]
    pats = MLA_PAT
    if "--kernel" in args:
        i = args.index("--kernel")
        pats = args[i + 1]
        args = args[:i] + args[i + 2:]
    main(args[0] if args else "/shared_nfs/kk/pr35619/trace_c128_pdi24_steady",
         pats.split(","))
