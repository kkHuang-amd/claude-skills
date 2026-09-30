"""Per-request attribution for AgentX c>=2: how much of each request's decode window overlaps another
request's prefill (its [start, start+TTFT] window), vs its interactivity.
  python3 agentx_prefill_overlap.py <result_dir>"""
import json, sys
import numpy as np

D = sys.argv[1]
R = []
for line in open(f"{D}/aiperf_artifacts/profile_export.jsonl"):
    d = json.loads(line)
    m = d["metrics"]; g = lambda k: (m.get(k) or {}).get("value")
    if g("time_to_first_token") is None:
        continue
    s = d["metadata"]["request_start_ns"] / 1e9
    R.append(dict(prof=d["metadata"].get("benchmark_phase") == "profiling", s=s, f=s + g("time_to_first_token") / 1e3,
                  e=s + g("request_latency") / 1e3, itl=g("inter_token_latency"), osl=g("output_sequence_length"),
                  isl=g("input_sequence_length"), depth=d["metadata"].get("agent_depth") or 0))
pre = [(r["s"], r["f"]) for r in R]
rows = []
for i, r in enumerate(R):
    if not r["prof"] or not r["itl"] or r["e"] <= r["f"]:
        continue
    ov = sum(max(0.0, min(r["e"], f) - max(r["f"], s)) for j, (s, f) in enumerate(pre) if j != i)
    inf = sum(max(0.0, min(r["e"], q["e"]) - max(r["f"], q["s"])) for j, q in enumerate(R) if j != i)
    dur = r["e"] - r["f"]
    rows.append((1000 / r["itl"], ov / dur, ov, r["osl"], 1 + inf / dur, r["depth"]))
a = np.array(rows)
print(f"n={len(a)}  intvty p50 {np.percentile(a[:,0],50):.1f}  P90(low) {np.percentile(a[:,0],10):.1f}")
print(f"requests with any overlap: {(a[:,1]>0).sum()}  corr(intvty, overlap_frac) {np.corrcoef(a[:,0], a[:,1])[0,1]:.3f}")
for lo, hi in [(0, 1e-9), (1e-9, 0.02), (0.02, 0.1), (0.1, 1e9)]:
    s = (a[:, 1] >= lo) & (a[:, 1] < hi) if lo > 0 else a[:, 1] == 0
    if s.sum():
        print(f"  overlap_frac [{lo:.2f},{hi:.2f}) n={s.sum():3d} intvty med {np.median(a[s,0]):.1f} p10 {np.percentile(a[s,0],10):.1f}")
no = a[a[:, 1] == 0, 0]
if len(no):
    print(f"P90 if no request overlapped (no-overlap subset P90): {np.percentile(no,10):.1f}")
low = a[a[:, 0] <= np.percentile(a[:, 0], 10)]
print(f"bottom-decile: n={len(low)} overlapped {(low[:,1]>0).sum()}  median overlap_frac {np.median(low[:,1]):.3f}  median osl {np.median(low[:,3]):.0f}")
# In-flight load: column 4 = time-weighted number of requests in flight during the decode window (self included;
# subagents do not hold a concurrency slot, so this can exceed CONC); column 5 = agent_depth.
print(f"in-flight during decode: p50 {np.median(a[:,4]):.2f} p90 {np.percentile(a[:,4],90):.2f} max {a[:,4].max():.2f}"
      f"  corr(intvty, inflight) {np.corrcoef(a[:,0], a[:,4])[0,1]:.3f}  corr(intvty, prefill_load) {np.corrcoef(a[:,0], a[:,1])[0,1]:.3f}")
for name, s in [("depth 0", a[:, 5] == 0), ("depth>0", a[:, 5] > 0)]:
    if s.sum():
        print(f"  {name}: n={s.sum():4d} intvty med {np.median(a[s,0]):.1f} p10 {np.percentile(a[s,0],10):.1f} inflight med {np.median(a[s,4]):.2f}")
for lo, hi in [(0, 4.5), (4.5, 8.5), (8.5, 12.5), (12.5, 1e9)]:
    s = (a[:, 4] >= lo) & (a[:, 4] < hi)
    if s.sum():
        n0 = s & (a[:, 1] == 0)
        print(f"  inflight [{lo:.1f},{hi:.1f}) n={s.sum():4d} intvty med {np.median(a[s,0]):.1f} p10 {np.percentile(a[s,0],10):.1f}"
              f"  prefill_load med {np.median(a[s,1]):.3f}  no-overlap n={n0.sum()} med {np.median(a[n0,0]) if n0.sum() else float('nan'):.1f}")
P = sorted([(r["s"], 1) for r in R if r["prof"]] + [(r["e"], -1) for r in R if r["prof"]])
t0, cur, occ = P[0][0], 0, {}
for t, dlt in P:
    occ[cur] = occ.get(cur, 0) + t - t0; cur += dlt; t0 = t
T = sum(occ.values())
print("wall-time share by in-flight count: " + " ".join(f"{k}:{v/T:.2f}" for k, v in sorted(occ.items()) if v / T >= 0.01)
      + f"  (>{max(8, 0)}: {sum(v for k, v in occ.items() if k > 8)/T:.2f}, max {max(occ)})")
