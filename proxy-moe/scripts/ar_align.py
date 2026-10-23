"""Align all-reduce kernels across ranks by host timestamp (not call index) and split each
AR into arrival skew vs own cost.

Usage: python3 ar_align.py <dir with *TP-<r>*.trace.json.gz> [--pattern REGEX]
For every TP0 AR, take on each other rank the AR whose interval overlaps it most.
  skew = last start - first start    (time early ranks spend waiting for the last one)
  cost = earliest end - last start   (the AR after everyone has arrived)
Assumes the per-rank traces share a host clock (one node).
"""
import argparse, bisect, glob, gzip, json, re, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--pattern", default=r"nccl|all_reduce|allreduce|cross_device")
a = ap.parse_args()
pat = re.compile(a.pattern, re.I)

ars = {}
for f in glob.glob(f"{a.dir}/**/*.trace.json.gz", recursive=True):
    r = int(re.search(r"TP-(\d+)", f).group(1))
    ev = json.load(gzip.open(f))["traceEvents"]
    ars[r] = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
                    if e.get("ph") == "X" and e.get("cat") == "kernel" and pat.search(e["name"]))

def best(lst, s, e):
    i = bisect.bisect_left(lst, (s, s))
    cands = [lst[j] for j in (i - 1, i, i + 1) if 0 <= j < len(lst)]
    ov = [(min(e, ce) - max(s, cs), (cs, ce)) for cs, ce in cands]
    o, c = max(ov)
    return c if o > 0 else None

skews, costs, durs = [], [], []
for s, e in ars[0]:
    m = [best(ars[r], s, e) for r in sorted(ars)]
    if any(x is None for x in m):
        continue
    starts, ends = [x[0] for x in m], [x[1] for x in m]
    skews.append(max(starts) - min(starts))
    costs.append(min(ends) - max(starts))
    durs.append(e - s)
n = len(costs)
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print(f"aligned {n}/{len(ars[0])} TP0 ARs across {len(ars)} ranks")
print(f"TP0 AR duration us: med {st.median(durs):.1f}  p90 {q(durs, .9):.1f}")
print(f"arrival skew us:    med {st.median(skews):.1f}  p90 {q(skews, .9):.1f}")
print(f"own cost us:        med {st.median(costs):.1f}  p10 {q(costs, .1):.1f}  p90 {q(costs, .9):.1f}")
print(f"sum per AR call:    TP0 dur {sum(durs)/n:.1f}  = skew {sum(skews)/n:.1f} + cost {sum(costs)/n:.1f} (approx)")
