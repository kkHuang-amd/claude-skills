"""Per-step decode kernel breakdown from a torch profiler chrome trace (.json.gz).

The profiler window of clk_profile_1029.sh also holds prefill/warmup steps, so dividing the whole
trace by --steps (trace_kernels.py) mixes them in. This splits the GPU timeline into steps at the
sampling ArgMax kernel (one per forward), keeps steps whose kernel set looks like decode
(no kernel matching --prefill-pat, and the gate-up count == --layers), and reports per-step
median wall / busy / per-category / per-kernel times over those steps.

Usage: python3 decode_steps.py <trace.json.gz> [--layers 12] [--top 20] [--dump-steps]
"""

import argparse
import gzip
import json
import re
import statistics as st
from collections import defaultdict

from trace_kernels import cat

STEP_PAT = r"ArgMaxOps"
GATEUP_PAT = r"a8w4.*K16384"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--prefill-pat", default=r"ncclDevKernel_Generic|Cijk|noKS|extend|prefill|fmha|unified_attention")
    ap.add_argument("--dump-steps", action="store_true")
    a = ap.parse_args()
    with gzip.open(a.trace, "rt") as f:
        ev = json.load(f)["traceEvents"]
    ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                key=lambda e: e["ts"])
    steps, cur = [], []
    for e in ks:
        cur.append(e)
        if re.search(STEP_PAT, e["name"]):
            steps.append(cur)
            cur = []
    keep = []
    for i, s in enumerate(steps):
        names = [e["name"] for e in s]
        pre = any(re.search(a.prefill_pat, n) for n in names)
        gu = sum(1 for n in names if re.search(GATEUP_PAT, n))
        wall = (s[-1]["ts"] + s[-1]["dur"] - s[0]["ts"]) / 1000
        busy = sum(e["dur"] for e in s) / 1000
        if a.dump_steps:
            print(f"step {i:3d} n={len(s):4d} gateup={gu:2d} prefill={int(pre)} wall={wall:7.2f} busy={busy:7.2f} ms")
        if not pre and gu == a.layers:
            keep.append(s)
    if not keep:
        print(f"no decode steps found ({len(steps)} steps total)")
        return
    # wall = first kernel start of this step -> first kernel start of the next step (includes the CPU gap)
    gaps = []
    for s in keep:
        idx = steps.index(s)
        if idx + 1 < len(steps):
            gaps.append((steps[idx + 1][0]["ts"] - s[0]["ts"]) / 1000)
    busy = [sum(e["dur"] for e in s) / 1000 for s in keep]
    print(f"decode steps {len(keep)}/{len(steps)}  step period med {st.median(gaps):.2f} ms  "
          f"GPU busy med {st.median(busy):.2f} ms  (idle {100 * (1 - st.median(busy) / st.median(gaps)):.1f}%)")
    per_cat = defaultdict(list)
    per_k = defaultdict(list)
    for s in keep:
        c, k = defaultdict(float), defaultdict(float)
        for e in s:
            c[cat(e["name"])] += e["dur"]
            k[e["name"]] += e["dur"]
        for n in set(cat(e["name"]) for e in ks):
            per_cat[n].append(c.get(n, 0.0))
        for n in k:
            per_k[n].append(k[n])
    tb = st.median(busy) * 1000
    print("by category (median us/step): " + "  ".join(
        f"{n}={st.median(v):.0f}({100 * st.median(v) / tb:.1f}%)"
        for n, v in sorted(per_cat.items(), key=lambda x: -st.median(x[1])) if st.median(v) > 0))
    cnt = defaultdict(int)
    for e in keep[0]:
        cnt[e["name"]] += 1
    for n, v in sorted(per_k.items(), key=lambda x: -st.median(x[1]))[: a.top]:
        m = st.median(v) if len(v) == len(keep) else sum(v) / len(keep)
        print(f"{100 * m / tb:5.1f}% {m:8.1f}us n={cnt.get(n, 0):3d} {m / max(cnt.get(n, 1), 1):7.1f}us/call [{cat(n)}] {n[:100]}")


if __name__ == "__main__":
    main()
