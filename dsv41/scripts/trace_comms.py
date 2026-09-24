#!/usr/bin/env python3
"""List collectives in a torch trace: record_param_comms CPU ops (name, dtype, msg size) + total NCCL kernel time.
Usage: trace_comms.py <trace.json.gz>"""
import gzip, json, sys
from collections import defaultdict
t = json.load(gzip.open(sys.argv[1])); ev = t.get("traceEvents", [])
agg = defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("ph") == "X" and e.get("name") == "record_param_comms":
        a = e.get("args", {}); k = (a.get("Collective name"), a.get("dtype"), a.get("In msg nelems"), a.get("Out msg nelems"), a.get("Group size"))
        agg[k][0] += 1; agg[k][1] += e.get("dur", 0)
print("collective, dtype, in_nelems, out_nelems, group | count | cpu_ms")
for k, (n, d) in sorted(agg.items(), key=lambda x: -x[1][0])[:15]: print(f"  {k} | {n} | {d/1e3:.1f}")
nk = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower()]
durs = sorted(e["dur"] for e in nk)
if durs: print(f"nccl kernels={len(durs)} total={sum(durs)/1e3:.1f} ms  min={durs[0]/1e3:.2f} p50={durs[len(durs)//2]/1e3:.2f} max={durs[-1]/1e3:.2f} ms")
