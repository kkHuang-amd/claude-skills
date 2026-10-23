"""Where does all-reduce wait come from? Compare the work BETWEEN consecutive ARs across ranks.

Usage: python3 ar_segment_skew.py <dir with *TP-<r>*.trace.json.gz> --pattern <AR kernel regex> [--steps 20]
For segment i (kernels after AR i-1 ends and before AR i starts) on each rank: busy = summed kernel time,
span = AR(i).start - AR(i-1).end (busy + GPU idle gaps). Matched by AR index across ranks.
If max-minus-mean of busy is small but of span is large, the skew is GPU idle (host/launch/arrival),
not kernel work. Per-kernel-name spread inside segments ranks which kernel varies most across ranks.
"""
import argparse, collections, glob, gzip, json, re, statistics as st
from concurrent.futures import ProcessPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--pattern", required=True)
ap.add_argument("--steps", type=int, default=20)
a = ap.parse_args()


def load(f):
    ev = [e for e in json.load(gzip.open(f, "rt"))["traceEvents"]
          if e.get("ph") == "X" and e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    segs, cur, last_end = [], [], None
    for e in ev:
        if re.search(a.pattern, e["name"]):
            if last_end is not None:
                segs.append((sum(d for _, d in cur), e["ts"] - last_end, cur))
            cur, last_end = [], e["ts"] + e["dur"]
        elif last_end is not None:
            cur.append((e["name"][:60], e["dur"]))
    return int(re.search(r"TP-(\d+)", f).group(1)), segs


with ProcessPoolExecutor(8) as ex:
    R = dict(ex.map(load, glob.glob(a.dir + "/*TP-*.trace.json.gz")))
n = min(len(v) for v in R.values())
busy_sp, span_sp = [], []
name_sp = collections.defaultdict(float)
for i in range(n):
    b = [R[r][i][0] for r in R]; s = [R[r][i][1] for r in R]
    busy_sp.append(max(b) - st.mean(b)); span_sp.append(max(s) - st.mean(s))
    per = collections.defaultdict(list)
    for r in R:
        agg = collections.defaultdict(float)
        for nm, d in R[r][i][2]:
            agg[nm] += d
        for nm, d in agg.items():
            per[nm].append(d)
    for nm, v in per.items():
        if len(v) == len(R):
            name_sp[nm] += max(v) - st.mean(v)
print(f"ranks={len(R)} segments={n}")
print(f"per step: busy max-mean {sum(busy_sp)/a.steps/1e3:.2f} ms, span max-mean {sum(span_sp)/a.steps/1e3:.2f} ms")
print(f"per segment median: busy {st.median(busy_sp):.1f} us, span {st.median(span_sp):.1f} us")
print("kernels with the largest cross-rank spread (ms/step, summed max-mean):")
for nm, v in sorted(name_sp.items(), key=lambda x: -x[1])[:8]:
    print(f"  {v/a.steps/1e3:6.3f}  {nm}")
