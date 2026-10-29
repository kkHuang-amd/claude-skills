# Top-N AR instances by arrival spread (end-time aligned), with fastest/slowest rank and times relative to TP0 trace start.
import glob, gzip, json, re, sys, bisect, statistics as st
from concurrent.futures import ProcessPoolExecutor
PAT = re.compile(r"(?i)allreduce|ar_ll")
def load(f):
    ev = sorted((e for e in json.load(gzip.open(f, "rt"))["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
    return int(re.search(r"TP-(\d+)", f).group(1)), ([(e["ts"], e["ts"] + e["dur"]) for e in ev if PAT.search(e["name"])], ev[0]["ts"])
d, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
with ProcessPoolExecutor(8) as ex: R = dict(ex.map(load, glob.glob(d + "/*TP-*.trace.json.gz")))
ranks = sorted(R); ends = {r: [x[1] for x in R[r][0]] for r in ranks}; t0 = R[0][1]
out = []
for i, (s0, e0) in enumerate(R[0][0]):
    arr = {}
    for r in ranks:
        j = bisect.bisect_left(ends[r], e0); j = min((k for k in (j - 1, j) if 0 <= k < len(ends[r])), key=lambda k: abs(ends[r][k] - e0))
        if abs(ends[r][j] - e0) > 30: break
        arr[r] = R[r][0][j][0]
    else:
        fr, sr = min(arr, key=arr.get), max(arr, key=arr.get)
        out.append((arr[sr] - arr[fr], i, fr, sr, e0))
med = st.median(x[0] for x in out)
print(f"AR instances {len(out)}, arrival max-min median {med:.0f} us; AR index i = i-th AR on TP0 (25 per step, step = i//25)")
for sp, i, fr, sr, e0 in sorted(out, reverse=True)[:N]:
    print(f"AR#{i:3d} (step {i//25:2d}, AR {i%25:2d} in step)  spread {sp:5.0f} us  fastest TP{fr}  slowest TP{sr}  AR ends at +{(e0-t0)/1e3:.3f} ms from TP0 first kernel")
