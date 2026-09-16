#!/usr/bin/env python3
"""Is a MoE kernel a cross-rank BARRIER or real work? Falsification by trace.

    python3 prepare_wait.py <trace dir>

Motivation: `megamoe_prepare_compact` is 16.24 ms/step on MI355X at grid 30 of
256 CUs, and reading the generator shows it is a cross-rank dispatch protocol
with an explicit spin-wait (`comm_ops.wait_i32_until_equals` on a system-scope
epoch gate, `mega_moe_prepare.py:188`). Source shows the MECHANISM exists; it
does not show how much of the 266 us/call is spent in it.

The test: a barrier's duration ANTI-correlates with the rank's own work -- a
busy rank arrives late and waits less -- while real work tracks own work. This
is the same logic `trace_ranks.py` applies to the `barrier` bucket, narrowed to
individual kernels.

Reads `compute` (attn+gemm+quant+norm_rope+sample) as the proxy for own work,
since MI355X's `compute` is not pace-pinned (see FINDINGS) and the verify steps
carry no token annotation.

Caveat: each rank sits at its own `bs`, so rank and batch move together. Rank 0
spans several `bs` values on its own and is the within-rank control.
"""
import collections
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import attribute, classify, load, rank_of, verify_classes  # noqa: E402

COMPUTE = ("attn", "gemm", "quant", "norm_rope", "sample")
# Keep the two MLA decode variants SEPARATE. aiter picks
# `_paged_decode_split_kernel` at small bs and `_paged_decode_fused_kernel` at
# larger bs, so a pattern matching both (or only one) mixes heterogeneous
# kernels into one µs/call average and the correlation becomes meaningless.
# Rows where a kernel is absent are excluded from its correlation, not read as 0.
TRACK = {
    # ROCm / aiter
    "prepare": "megamoe_prepare_compact",
    "stage1": "megamoe_stage1_compact",
    "stage2": "megamoe_stage2_compact",
    "ep_combine": "ep_combine_intranode",
    "mla_split": "_paged_decode_split_kernel",
    "mla_fused": "_paged_decode_fused_kernel",
    # CUDA / deep_gemm -- so B200 can run this file unchanged. Its wait has no
    # dedicated kernel: it is absorbed inside the fused mega_moe_impl, which is
    # exactly what `b200_moe` below tests.
    "b200_moe": "mega_moe_impl",
    "b200_gemm": "gemm_1d1d_impl",
    "b200_mla": "flash_fwd_splitkv_mla",
}


def main(d):
    rows = []
    for f in sorted(Path(d).glob("*.trace.json.gz"), key=rank_of):
        steps, kernels = load(f)
        owned = attribute(steps, kernels)
        vc = verify_classes(steps, owned)
        per_bs = collections.defaultdict(list)
        for i, s in enumerate(steps):
            if s["type"] == "TARGET_VERIFY" and vc.get(i) == " full":
                per_bs[s["bs"]].append(owned.get(i, []))
        for bs, ents in per_bs.items():
            n = len(ents)
            comp = []
            per_k = collections.defaultdict(lambda: [0.0, 0])
            for ks in ents:
                acc = collections.defaultdict(float)
                for dur, name in ks:
                    acc[classify(name)] += dur / 1000.0
                    for tag, pat in TRACK.items():
                        if pat in name:
                            per_k[tag][0] += dur
                            per_k[tag][1] += 1
                comp.append(sum(acc[r] for r in COMPUTE))
            rows.append((rank_of(f), bs, n, st.median(comp),
                         {t: (v[0] / max(v[1], 1), v[1] / n) for t, v in per_k.items()}))

    if not rows:
        print("no full TARGET_VERIFY steps found")
        return
    rows.sort(key=lambda r: r[3])
    print(f"=== {d}\nsorted by the rank's own compute (us/call for each kernel)\n")
    print(f"{'rank':>4s} {'bs':>3s} {'n':>3s} {'compute':>8s} | "
          + " ".join(f"{t:>11s}" for t in TRACK))
    for rank, bs, n, comp, k in rows:
        print(f"{rank:4d} {bs:3d} {n:3d} {comp:8.2f} | "
              + " ".join(f"{k.get(t, (0, 0))[0]:11.1f}" for t in TRACK))
    print(f"\ncalls/step: "
          + " ".join(f"{t}={rows[-1][4].get(t, (0, 0))[1]:.0f}" for t in TRACK))

    print("\ncorrelation of each kernel's us/call against the rank's own compute"
          "\n(rows where the kernel is absent are excluded, never read as 0):")
    for t in TRACK:
        pairs = [(r[3], r[4][t][0]) for r in rows
                 if t in r[4] and r[4][t][1] > 0]
        if len(pairs) < 3:
            print(f"  {t:<12s} only {len(pairs)} rank/bs cells -- not enough to correlate")
            continue
        comps = [c for c, _ in pairs]
        v = [x for _, x in pairs]
        mc, mv = st.fmean(comps), st.fmean(v)
        num = sum((c - mc) * (x - mv) for c, x in zip(comps, v))
        den = (sum((c - mc) ** 2 for c in comps)
               * sum((x - mv) ** 2 for x in v)) ** 0.5
        r = num / den if den else 0.0
        verdict = ("ANTI-correlated -> it is a WAIT" if r < -0.5 else
                   "tracks own work -> real work" if r > 0.5 else
                   "no clear trend -- inconclusive")
        print(f"  {t:<12s} n={len(pairs)} r={r:+.3f}  "
              f"spread {min(v):.1f}-{max(v):.1f} us  {verdict}")
    print("\nReminder: rank and bs move together here; treat a single-rank, "
          "multi-bs row group (rank 0) as the controlled comparison.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/shared_nfs/kk/pr35619/trace_c128_pdi24_steady")
