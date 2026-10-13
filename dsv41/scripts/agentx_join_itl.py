#!/usr/bin/env python3
"""Join two AgentX aiperf runs per request (conversation_id, turn_index; profiling phase) and bin the ITL ratio by ISL.
Constant ratio across ISL bins = per-step fixed cost (layers / draft / host); ratio growing with ISL = context-dependent
(indexer / attention). Usage: agentx_join_itl.py <run_dir_A> <run_dir_B>   (dirs holding aiperf_artifacts/)
Prints per ISL bin: n, median ITL A / B (ms), median per-request B/A ratio, and a least-squares ITL = a + b*ISL fit."""
import json, statistics as st, sys


def load(d):
    out = {}
    for l in open(f"{d}/aiperf_artifacts/profile_export.jsonl"):
        r = json.loads(l); m, md = r["metrics"], r["metadata"]
        if md.get("benchmark_phase") != "profiling" or "inter_token_latency" not in m:
            continue
        if m.get("output_sequence_length", {}).get("value", 0) < 16:
            continue
        out[(md["conversation_id"], md["turn_index"])] = (m["input_sequence_length"]["value"], m["inter_token_latency"]["value"],
                                                         m["output_sequence_length"]["value"])
    return out


def fit(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


A, B = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(A) & set(B), key=lambda k: A[k][0])
print(f"A={len(A)} B={len(B)} joined={len(keys)}")
edges = [0, 32e3, 64e3, 128e3, 256e3, 512e3, 2e6]
for lo, hi in zip(edges, edges[1:]):
    ks = [k for k in keys if lo <= A[k][0] < hi]
    if not ks:
        continue
    print(f"ISL [{lo/1e3:>4.0f}k,{hi/1e3:>5.0f}k) n={len(ks):3d}  ITL A {st.median(A[k][1] for k in ks):6.3f}  "
          f"B {st.median(B[k][1] for k in ks):6.3f}  B/A {st.median(B[k][1] / A[k][1] for k in ks):5.3f}")
for name, R in (("A", A), ("B", B)):
    a, b = fit([(R[k][0], R[k][1]) for k in keys])
    print(f"fit {name}: ITL_ms = {a:.3f} + {b * 1e5:.4f} per 100k ISL")
print(f"all: median B/A {st.median(B[k][1] / A[k][1] for k in keys):.3f}")
