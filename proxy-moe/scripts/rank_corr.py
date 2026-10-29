# Cross-rank correlation of per-call durations of one kernel (matched by call index). Usage: rank_corr.py <trace dir> <kernel regex>
import glob, gzip, json, re, sys, statistics as st
d, pat = sys.argv[1], re.compile(sys.argv[2])
dur = {}
for f in glob.glob(f"{d}/*TP-*.trace.json.gz"):
    r = int(re.search(r"TP-(\d+)", f).group(1))
    ev = sorted((e for e in json.load(gzip.open(f))["traceEvents"] if e.get("cat") == "kernel" and pat.search(e["name"])), key=lambda e: e["ts"])
    dur[r] = [e["dur"] for e in ev]
n = min(len(v) for v in dur.values()); R = sorted(dur)
print(f"calls matched {n}; TP0 us p10/p50/p90 = " + "/".join(f"{q:.0f}" for q in [st.quantiles(dur[0][:n], n=10)[i] for i in (0, 4, 8)]))
c = lambda a, b: st.correlation(dur[a][:n], dur[b][:n])
same = [c(a, a + 1) for a in range(0, 8, 2)]
cross = [c(a, b) for a in R for b in R if a < b and a // 2 != b // 2]
print("same-socket pairs:", " ".join(f"{x:.2f}" for x in same))
print(f"cross-socket pairs: min {min(cross):.2f} median {st.median(cross):.2f} max {max(cross):.2f} (n={len(cross)})")
print("per-rank mean us:", " ".join(f"TP{r}={st.mean(dur[r][:n]):.0f}" for r in R))
