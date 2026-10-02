"""Paired per-request comparison of two AgentX runs (same trace => same requests).

Requests are matched by (conversation_id, turn_index); only profiling-phase, non-cancelled
records with >1 output token. Removes the request-mix noise that moves aggregate
tok/s/GPU and interactivity between runs.

usage: paired_itl.py <base result dir> <cand result dir> [...]
"""
import json, statistics as st, sys

def load(d):
    out = {}
    for line in open(f"{d}/aiperf_artifacts/profile_export.jsonl"):
        r = json.loads(line); m, md = r.get("metrics", {}), r["metadata"]
        if md.get("benchmark_phase") != "profiling" or md.get("was_cancelled"):
            continue
        itl = m.get("inter_token_latency", {}).get("value")
        osl = m.get("output_sequence_length", {}).get("value", 0)
        if not itl or osl <= 1:
            continue
        out[(md["conversation_id"], md["turn_index"])] = (itl, m.get("input_sequence_length", {}).get("value"))
    return out

def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]

base = load(sys.argv[1])
print(f"base {sys.argv[1].rstrip('/').split('/')[-1]}: {len(base)} requests")
for d in sys.argv[2:]:
    c = load(d); keys = sorted(base.keys() & c.keys())
    r = [c[k][0] / base[k][0] for k in keys]
    bi = [1000 / base[k][0] for k in keys]; ci = [1000 / c[k][0] for k in keys]
    print(f"{d.rstrip('/').split('/')[-1]}: {len(c)} requests, {len(keys)} matched")
    print(f"  ITL ratio cand/base: median {st.median(r):.4f}  mean {st.mean(r):.4f}  p10 {pct(r, .1):.3f}  p90 {pct(r, .9):.3f}")
    # aiperf's "intvty p90" is the slow tail: 1 / (p90 of ITL) == p10 of 1/ITL.
    for name, b, cc in (("matched", bi, ci), ("all", [1000 / v[0] for v in base.values()], [1000 / v[0] for v in c.values()])):
        print(f"  {name:7s} intvty slow-tail(aiperf p90) {pct(b, .1):.1f} -> {pct(cc, .1):.1f} ({pct(cc, .1) / pct(b, .1) - 1:+.1%})"
              f"   p50 {pct(b, .5):.1f} -> {pct(cc, .5):.1f} ({pct(cc, .5) / pct(b, .5) - 1:+.1%})")
