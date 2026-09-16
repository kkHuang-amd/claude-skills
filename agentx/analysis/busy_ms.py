#!/usr/bin/env python3
"""Elapsed GPU time per role, not summed kernel duration. RUN THIS ON BOTH NODES.

    python3 busy_ms.py <trace dir or file> [bs]

`trace_summary.py` and `kernel_dump.py` report a SUM of kernel durations. On a
platform with stream concurrency that sum double-counts: B200 measured 50.86 ms
of summed kernel inside a 30.0 ms step (1.67x), so its per-role sums cannot be
subtracted from MI355X's, whose sum equals its wall (1.00x). Comparing sums
silently compares "kernel-seconds issued" on one node against "elapsed time" on
the other.

What is comparable is ELAPSED time. This tool sweeps the kernel intervals and
reports three things per role:

  union      time during which at least one kernel of this role was running
  exclusive  time during which ONLY this role was running
  credited   exclusive + a 1/k share of every segment where k roles overlap

`credited` is the useful column: the credits sum exactly to total GPU busy time,
so a cross-platform per-role table built from `credited` is an attribution of
elapsed time and the deltas mean what they look like. `union` per role does NOT
sum to total busy whenever roles overlap each other.

Also prints total busy vs step wall, i.e. how much of the step the GPU was idle
-- a number neither node has ever published.
"""
import collections
import gzip
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import GPU_CATS, classify, load, rank_of, verify_classes  # noqa: E402


def raw_kernels(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        ev = json.load(f)["traceEvents"]
    return sorted((e["ts"], e["dur"], e["name"]) for e in ev
                  if e.get("ph") == "X" and e.get("cat") in GPU_CATS and e.get("dur"))


def sweep(intervals):
    """intervals: [(lo, hi, role)] in us.
    -> (busy_ms, {role: exclusive_ms}, {role: credited_ms}, {role: union_ms})"""
    edges = []
    for lo, hi, role in intervals:
        edges.append((lo, 1, role))
        edges.append((hi, -1, role))
    edges.sort(key=lambda e: (e[0], -e[1]))

    active = collections.Counter()
    busy = 0.0
    excl = collections.defaultdict(float)
    cred = collections.defaultdict(float)
    uni = collections.defaultdict(float)
    prev = None
    for ts, delta, role in edges:
        live = {r for r, c in active.items() if c > 0}
        if prev is not None and ts > prev and live:
            seg = (ts - prev) / 1000.0
            busy += seg
            for r in live:
                uni[r] += seg
            if len(live) == 1:
                excl[next(iter(live))] += seg
            for r in live:
                cred[r] += seg / len(live)
        active[role] += delta
        prev = ts
    return busy, excl, cred, uni


def steps_of_interest(path, want_bs):
    steps, _ = load(path)
    ks = raw_kernels(path)
    owned = collections.defaultdict(list)
    for ts, dur, name in ks:
        for i, s in enumerate(steps):
            if s["lo"] <= ts <= s["hi"]:
                owned[i].append((ts, dur, name))
                break
    vc = verify_classes(steps, {i: [(d, n) for _t, d, n in v] for i, v in owned.items()})
    out = []
    for i, s in enumerate(steps):
        if s["type"] != "TARGET_VERIFY" or vc.get(i) != " full":
            continue
        if want_bs is not None and s["bs"] != want_bs:
            continue
        out.append((s, owned.get(i, [])))
    return out


def report(path, want_bs):
    ents = steps_of_interest(path, want_bs)
    if not ents:
        print(f"  {Path(path).name[-28:]}: no full TARGET_VERIFY"
              + (f" at bs={want_bs}" if want_bs is not None else ""))
        return
    bss = sorted({s["bs"] for s, _ in ents})
    walls, sums, busies = [], [], []
    cred_acc = collections.defaultdict(list)
    excl_acc = collections.defaultdict(list)
    uni_acc = collections.defaultdict(list)
    sum_acc = collections.defaultdict(list)
    for s, kk in ents:
        walls.append(s["ms"])
        sums.append(sum(d for _t, d, _n in kk) / 1000.0)
        busy, excl, cred, uni = sweep([(t, t + d, classify(n)) for t, d, n in kk])
        busies.append(busy)
        rsum = collections.defaultdict(float)
        for _t, d, n in kk:
            rsum[classify(n)] += d / 1000.0
        for r in set(list(cred) + list(excl) + list(uni) + list(rsum)):
            cred_acc[r].append(cred.get(r, 0.0))
            excl_acc[r].append(excl.get(r, 0.0))
            uni_acc[r].append(uni.get(r, 0.0))
            sum_acc[r].append(rsum.get(r, 0.0))

    print(f"\n  rank {rank_of(path)}  bs={bss}  n={len(ents)}")
    w, sm, bz = st.fmean(walls), st.fmean(sums), st.fmean(busies)
    print(f"    step wall      {w:8.2f} ms   (p50 {st.median(walls):.2f})")
    print(f"    summed kernel  {sm:8.2f} ms   sum/busy {sm / max(bz, 1e-9):.3f}x"
          f"  <-- do NOT compare this across platforms")
    print(f"    GPU busy       {bz:8.2f} ms   busy/wall {bz / max(w, 1e-9):.3f}"
          f"   idle in step {w - bz:.2f} ms")
    print(f"    {'role':<12s} {'sum':>8s} {'union':>8s} {'excl':>8s} {'credited':>9s}")
    for r in sorted(cred_acc, key=lambda r: -st.fmean(cred_acc[r])):
        print(f"    {r:<12s} {st.fmean(sum_acc[r]):8.2f} {st.fmean(uni_acc[r]):8.2f}"
              f" {st.fmean(excl_acc[r]):8.2f} {st.fmean(cred_acc[r]):9.2f}")
    print(f"    {'(totals)':<12s} {sum(st.fmean(v) for v in sum_acc.values()):8.2f}"
          f" {'':>8s} {'':>8s} {sum(st.fmean(v) for v in cred_acc.values()):9.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])
    want_bs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    files = sorted(target.glob("*.trace.json.gz"), key=rank_of) if target.is_dir() else [target]
    print(f"=== {target}"
          + (f"  bs={want_bs}" if want_bs is not None else "  (all bs)"))
    for f in files:
        report(f, want_bs)
