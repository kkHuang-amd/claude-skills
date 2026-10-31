#!/usr/bin/env python3
"""Per-layer-kind op table of VERIFY steps: kind = KDA / DSA / head / tail.
Segments end at each MoE stage2 (trace_layer_seq.py convention).
Usage: k31_layer_ops.py <trace.json.gz> [STEP=VERIFY] [nsteps=20]"""
import gzip, json, re, sys, statistics as st
from collections import defaultdict

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
ST = sys.argv[2] if len(sys.argv) > 2 else "VERIFY"
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
steps = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
               if e.get("cat") == "gpu_user_annotation" and e["name"].startswith(f"step[{ST}"))
ks_all = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                key=lambda e: e["ts"])
DELIM = re.compile(r"opus_moe_stage2|moe_stage2|down_ws_kernel")


def short(n):
    n = re.sub(r"<.*", "", n)
    n = re.sub(r"^void ", "", n)
    m = re.match(r"_ZN\d+(\w+?)\d+(\w+)", n)
    if n.startswith("_ZN5aiter"):
        n = "aiter::" + re.sub(r"^\d+", "", n[9:]).split("I")[0][:40]
    return n[:70]


def kind(seg):
    names = " ".join(e["name"] for e in seg)
    if "opus_mla_decode" in names or "mla_decode" in names:
        return "DSA"
    if "fused_sigmoid_gating" in names or "kda" in names.lower():
        return "KDA"
    return "OTHER"


agg = defaultdict(lambda: defaultdict(list))   # kind -> name -> [us per segment]
cnt = defaultdict(lambda: defaultdict(list))
seg_tot = defaultdict(list)
seg_n = defaultdict(list)
step_tot = []
use = steps[5:5 + NS] if len(steps) > 5 + NS else steps
for a, b in use:
    ks = [e for e in ks_all if a <= e["ts"] <= b]
    step_tot.append(sum(e["dur"] for e in ks))
    segs, cur = [], []
    for e in ks:
        cur.append(e)
        if DELIM.search(e["name"]):
            segs.append(cur); cur = []
    segs.append(cur)
    for i, s in enumerate(segs):
        k = "head(seg0)" if i == 0 else "tail" if i == len(segs) - 1 else kind(s)
        seg_tot[k].append(sum(e["dur"] for e in s)); seg_n[k].append(len(s))
        d, c = defaultdict(float), defaultdict(int)
        for e in s:
            d[short(e["name"])] += e["dur"]; c[short(e["name"])] += 1
        for n in set(list(d) + list(agg[k])):
            agg[k][n].append(d.get(n, 0.0)); cnt[k][n].append(c.get(n, 0))

print(f"{ST}: {len(use)} steps, kernel-sum/step {st.mean(step_tot)/1e3:.2f} ms")
for k in seg_tot:
    nseg = len(seg_tot[k]) / len(use)
    print(f"\n== {k}: {nseg:.1f} segs/step, {st.mean(seg_n[k]):.1f} kernels, {st.mean(seg_tot[k]):.1f} us/seg, "
          f"{st.mean(seg_tot[k]) * nseg / 1e3:.2f} ms/step")
    rows = sorted(agg[k].items(), key=lambda kv: -st.mean(kv[1]))
    for n, v in rows:
        m = st.mean(v)
        if m < 0.5:
            continue
        print(f"  {m:7.1f} us  x{st.mean(cnt[k][n]):4.1f}  {n}")
