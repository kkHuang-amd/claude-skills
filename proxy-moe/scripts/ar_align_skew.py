# AR-instance alignment by END timestamp (an AR finishes ~together on all ranks), then per segment (kernels between
# consecutive ARs) busy spread across ranks: max-mean, mean-min, max-min. Usage: ar_align_skew.py <trace dir> [steps]
import glob, gzip, json, re, sys, bisect, statistics as st
from concurrent.futures import ProcessPoolExecutor
PAT = re.compile(r"(?i)allreduce|ar_ll")
def load(f):
    ev = sorted((e for e in json.load(gzip.open(f, "rt"))["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
    ars, busy, cur = [], [], 0.0
    for e in ev:
        if PAT.search(e["name"]):
            ars.append((e["ts"], e["ts"] + e["dur"])); busy.append(cur); cur = 0.0
        else:
            cur += e["dur"]
    return int(re.search(r"TP-(\d+)", f).group(1)), (ars, busy)
d = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
with ProcessPoolExecutor(8) as ex: R = dict(ex.map(load, glob.glob(d + "/*TP-*.trace.json.gz")))
ranks = sorted(R); ends = {r: [x[1] for x in R[r][0]] for r in ranks}
rows, endspread = [], []
for i in range(1, len(R[0][0])):
    e0 = R[0][0][i][1]; idx = {}
    for r in ranks:
        j = bisect.bisect_left(ends[r], e0)
        j = min((k for k in (j - 1, j) if 0 <= k < len(ends[r])), key=lambda k: abs(ends[r][k] - e0))
        if abs(ends[r][j] - e0) > 30 or j == 0: break   # >30 us end mismatch = not the same AR instance
        idx[r] = j
    else:
        endspread.append(max(ends[r][idx[r]] for r in ranks) - min(ends[r][idx[r]] for r in ranks))
        b = [R[r][1][idx[r]] for r in ranks]                      # busy before this AR
        arr = [R[r][0][idx[r]][0] for r in ranks]                 # AR start = arrival
        rows.append((b, arr))
print(f"aligned AR instances {len(rows)} of {len(R[0][0])-1}; end-time spread median {st.median(endspread):.1f} us")
f = lambda fn: sum(fn(x) for x in (r[0] for r in rows)) / steps / 1e3
print(f"busy per step (ms): max-mean {f(lambda b: max(b)-st.mean(b)):.2f}  mean-min {f(lambda b: st.mean(b)-min(b)):.2f}  max-min {f(lambda b: max(b)-min(b)):.2f}")
g = lambda fn: sum(fn(x) for x in (r[1] for r in rows)) / steps / 1e3
print(f"arrival per step (ms): max-mean {g(lambda a: max(a)-st.mean(a)):.2f}  mean-min {g(lambda a: st.mean(a)-min(a)):.2f}  max-min {g(lambda a: max(a)-min(a)):.2f}")
last = [0] * 8; first = [0] * 8
for b, a in rows:
    last[ranks[a.index(max(a))]] += 1; first[ranks[a.index(min(a))]] += 1
print("last to arrive  :", " ".join(f"TP{r}={last[i]}" for i, r in enumerate(ranks)))
print("first to arrive :", " ".join(f"TP{r}={first[i]}" for i, r in enumerate(ranks)))
ab = [st.mean(a[i] - st.mean(a) for b, a in rows) for i in range(len(ranks))]
bb = [st.mean(b[i] - st.mean(b) for b, a in rows) for i in range(len(ranks))]
print("mean arrival offset vs rank-mean (us):", " ".join(f"TP{r}={ab[i]:+.0f}" for i, r in enumerate(ranks)))
print("mean busy offset vs rank-mean (us)   :", " ".join(f"TP{r}={bb[i]:+.0f}" for i, r in enumerate(ranks)))
