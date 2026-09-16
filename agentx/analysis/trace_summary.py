#!/usr/bin/env python3
"""Per-step-type GPU time breakdown for one or more SGLang chrome traces.

    python3 trace_summary.py <*.trace.json.gz> [...]

Output is deliberately stable and role-based so a CUDA trace and a ROCm trace
produce directly comparable tables (see trace_common.ROLES).

Two mistakes this tool exists to prevent, both made on real data:

1. Aggregating a whole file. `profile_by_stage` labels files `-DECODE` while
   they may hold a single `EXTEND` step, and one extend (~450 ms of GPU time)
   buries ten decode steps (~65 ms each). Steps are split by their `step[...]`
   annotation and NEVER mixed across types.
2. Reading a top-kernel list as a bottleneck ranking. In a DP-attention
   deployment the fused MoE all-to-all kernel absorbs the wait for the slowest
   rank, so on a lightly loaded rank it can be 90 % of the step while doing
   almost no work. `comm` is reported separately from compute for that reason,
   and cross-rank comparison (trace_ranks.py) is what attributes.
"""
import collections
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import attribute, classify, load, norm_name  # noqa: E402


def summarise(path, top=10):
    steps, kernels = load(path)
    owned = attribute(steps, kernels)

    print(f"\n=== {Path(path).name}")
    if not steps:
        print("  no step[...] annotations -- cannot attribute; refusing to aggregate blindly")
        return

    inv = collections.Counter(s["type"] for s in steps)
    print("  steps: " + ", ".join(f"{k}x{v}" for k, v in inv.most_common())
          + f"   (unattributed kernel time: "
            f"{sum(d for d, _ in owned.get(None, [])) / 1000:.1f} ms)")

    # Group by (type, bs), never by type alone. `bs` is forward_batch.batch_size
    # (profile_utils.py:477), so steps with different bs did different amounts
    # of work and must not be averaged together -- and two traces can only be
    # compared within matching bs buckets. Two B200 captures of the same arm
    # landed on bs=7 and bs=3 respectively, so this is not a corner case.
    by_type = collections.defaultdict(list)
    for i, s in enumerate(steps):
        key = f"{s['type']} bs={s['bs']}" if s["bs"] is not None else s["type"]
        by_type[key].append((s, owned.get(i, [])))

    for stype, entries in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        step_ms = [s["ms"] for s, _ in entries]
        toks = [s["toks"] for s, _ in entries if s["toks"] is not None]
        roles = collections.defaultdict(float)
        names = collections.defaultdict(lambda: [0.0, 0])
        for _, ks in entries:
            for dur, name in ks:
                roles[classify(name)] += dur
                slot = names[norm_name(name)]
                slot[0] += dur
                slot[1] += 1
        gpu = sum(roles.values()) or 1.0
        n = len(entries)
        # Per-step distributions, because per-step time is NOT stable. Measured
        # on 62 TARGET_VERIFY steps all at bs=7: moe 0.52-17.69 ms (34x), gemm
        # 0.61-20.63 ms (34x), attn 0.08-3.16 ms (40x) -- every role stretching
        # together, and role SHARES varying too (moe 20-62 % of the step). A
        # single step therefore proves nothing, and neither does a mean quoted
        # without its spread. Compare aggregates over many steps, with spread.
        per_step_roles = collections.defaultdict(list)
        for _, ks in entries:
            acc = collections.defaultdict(float)
            for dur, name in ks:
                acc[classify(name)] += dur
            for r in set(list(acc) + [k for k in roles]):
                per_step_roles[r].append(acc.get(r, 0.0) / 1000.0)

        print(f"\n  -- {stype}: n={n}  step ms p50={st.median(step_ms):.1f}"
              f"  mean={st.fmean(step_ms):.1f}"
              + (f"  tokens p50={st.median(toks):.0f} mean={st.fmean(toks):.0f}" if toks else ""))
        # Sum of kernel durations, NOT a busy fraction: kernels overlap across
        # streams, so this legitimately exceeds the step's wall time (measured
        # 25.7 ms of kernel inside a 16 ms step). Never divide it by wall time
        # and call the result utilisation.
        print(f"     GPU kernel ms per step (summed, overlaps): {gpu / 1000 / n:.2f}")
        print(f"     {'role':<12s} {'ms/step':>9s} {'%':>6s} {'min':>8s} {'p50':>8s} {'max':>8s}")
        for role in sorted(roles, key=lambda r: -roles[r]):
            v = sorted(per_step_roles.get(role) or [0.0])
            print(f"     {role:<12s} {roles[role] / 1000 / n:9.2f} {100 * roles[role] / gpu:6.1f}"
                  f" {v[0]:8.2f} {v[len(v) // 2]:8.2f} {v[-1]:8.2f}")
        print(f"     {'top kernels':<66s} {'ms/step':>9s} {'%':>6s} {'calls/step':>11s}")
        for name, (dur, c) in sorted(names.items(), key=lambda kv: -kv[1][0])[:top]:
            print(f"     {name:<66s} {dur / 1000 / n:9.2f} {100 * dur / gpu:6.1f} {c / n:11.1f}")
        unk = [(d, nm) for nm, (d, _) in names.items() if classify(nm) == "other" and d]
        if unk:
            worst = sorted(unk, reverse=True)[:3]
            print("     unclassified (add a ROLES pattern if these matter): "
                  + ", ".join(f"{nm[:34]} {d / 1000 / n:.2f}ms" for d, nm in worst))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        summarise(p)
