#!/usr/bin/env python3
"""Per-stream view of a decode torch-profiler trace: where multi-stream overlap comes from, and a kernel table for a
regex (default: mHC / norm / small elementwise).

    python3 -I trace_streams.py <trace.json.gz> --steps 40 [--match REGEX] [--top 25]

Streams = the `tid` of GPU kernel events. Overlap of a stream = its kernel time that runs while another stream also has
a kernel in flight. All numbers per decode step (window / --steps).
"""
import argparse
import collections
import gzip
import json
import re


def union(iv):
    tot = 0; cs = ce = None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if ce is not None else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace"); p.add_argument("--steps", type=int, default=40); p.add_argument("--top", type=int, default=25)
    p.add_argument("--match", default=r"mhc|hc_|sinkhorn|norm|rope|rotary|elementwise|vectorized|fill|copy|quant|cast")
    a = p.parse_args()
    ev = json.load(gzip.open(a.trace, "rt"))["traceEvents"]
    k = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"]
    S = a.steps
    by_s = collections.defaultdict(list)
    for e in k:
        by_s[e["tid"]].append(e)
    allv = [(e["ts"], e["ts"] + e["dur"]) for e in k]
    busy = union(allv); tot = sum(e["dur"] for e in k)
    print(f"kernels {len(k)}; kernel sum {tot / S / 1e3:.3f} ms/step, GPU busy (union) {busy / S / 1e3:.3f} ms/step, "
          f"overlap {(tot - busy) / S / 1e3:.3f} ms/step")
    print("stream, kernels/step, ms/step, ms/step overlapped with another stream, top kernels")
    for tid, es in sorted(by_s.items(), key=lambda x: -sum(e["dur"] for e in x[1])):
        other = [(e["ts"], e["ts"] + e["dur"]) for t2, l in by_s.items() if t2 != tid for e in l]
        mine = sum(e["dur"] for e in es)
        # overlapped time of this stream = |mine| + |other| - |mine U other|
        ov = mine + union(other) - union([(e["ts"], e["ts"] + e["dur"]) for e in es] + other) if other else 0
        top = collections.Counter()
        for e in es:
            top[e["name"][:50]] += e["dur"]
        tops = "; ".join(f"{n} {d / S:.0f}us" for n, d in top.most_common(3))
        print(f"  {tid}: {len(es) / S:6.1f}  {mine / S / 1e3:6.3f}  {ov / S / 1e3:6.3f}  {tops}")
    rx = re.compile(a.match, re.I)
    agg = collections.defaultdict(lambda: [0, 0.0, set()])
    for e in k:
        if rx.search(e["name"]):
            r = agg[e["name"]]; r[0] += 1; r[1] += e["dur"]; r[2].add(e["tid"])
    tot_m = sum(r[1] for r in agg.values())
    print(f"matching /{a.match}/: {tot_m / S / 1e3:.3f} ms/step in {sum(r[0] for r in agg.values()) / S:.0f} launches/step")
    print("  us/step  calls/step  us/call  streams  kernel")
    for n, (c, d, ss) in sorted(agg.items(), key=lambda x: -x[1][1])[:a.top]:
        print(f"  {d / S:7.1f}  {c / S:9.1f}  {d / c:7.2f}  {','.join(map(str, sorted(ss)))[:10]:>7}  {n[:95]}")


if __name__ == "__main__":
    main()
