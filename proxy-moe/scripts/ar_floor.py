"""Split all-reduce time into real cost vs waiting for slower ranks.

Usage: python3 ar_floor.py <dir with *TP-<r>*.trace.json.gz for all ranks> [--pattern REGEX]
Per rank: comm and non-comm kernel time. Per AR call (matched by index across ranks):
min over ranks ~= the AR's own cost (the last rank to arrive waits least); mean - min ~= skew wait.
"""
import argparse, glob, gzip, json, re, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--pattern", default=r"nccl|all_reduce|allreduce|cross_device")
a = ap.parse_args()
pat = re.compile(a.pattern, re.I)

ranks = {}
for f in glob.glob(f"{a.dir}/**/*.trace.json.gz", recursive=True):
    r = int(re.search(r"TP-(\d+)", f).group(1))
    ev = json.load(gzip.open(f))["traceEvents"]
    k = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"), key=lambda e: e["ts"])
    ranks[r] = ([e["dur"] for e in k if pat.search(e["name"])], sum(e["dur"] for e in k if not pat.search(e["name"])))

n = min(len(v[0]) for v in ranks.values())
print(f"ranks={len(ranks)} AR calls matched={n}")
for r in sorted(ranks):
    ar, other = ranks[r]
    print(f"TP{r}: AR {sum(ar)/1e3:8.2f} ms  non-AR {other/1e3:8.2f} ms  AR calls {len(ar)}")
mins = [min(ranks[r][0][i] for r in ranks) for i in range(n)]
means = [st.mean(ranks[r][0][i] for r in ranks) for i in range(n)]
print(f"per AR call (us): min-over-ranks median {st.median(mins):.1f}  mean-over-ranks median {st.median(means):.1f}")
print(f"sum over calls: min {sum(mins)/1e3:.2f} ms  mean {sum(means)/1e3:.2f} ms  -> skew/wait share {(1 - sum(mins)/sum(means))*100:.0f}%")
