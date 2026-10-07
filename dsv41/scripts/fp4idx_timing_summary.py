#!/usr/bin/env python3
"""Summarize idx_timing_rank0.jsonl (fp4idx_prefill_probe.sh, FP4_INDEX_PLANE_PORT.md P0.3b) per extend forward.

Spans: forward (model forward), indexer_extend (whole low-ratio indexer per layer), q_inputs, score (FP4 logits
call incl. its Python wrapper), ws_build (prefill workspace refresh; runs before the forward, so it is
attributed to fwd+1). gpu = time the stream took to pass the span (kernels + idle waiting for the host).
  python3 fp4idx_timing_summary.py <jsonl>
"""
import collections
import json
import statistics
import sys

recs = [json.loads(l) for l in open(sys.argv[1])]
fwd = {}
acc = collections.defaultdict(lambda: collections.defaultdict(float))
for r in recs:
    if r["name"] == "forward":
        fwd[r["fwd"]] = r
        continue
    f = r["fwd"] + 1 if r["name"] == "ws_build" else r["fwd"]
    a = acc[f]
    a[r["name"] + "_gpu"] += r["gpu_us"]
    a[r["name"] + "_host"] += r["host_us"]
    a[r["name"] + "_n"] += 1

hdr = (f"{'fwd':>4} {'tok':>6} {'reqs':>4} {'max_seq':>7} | {'fwd ms':>7} | {'idx ms':>7} {'idx%':>5} | "
       f"{'score ms':>8} {'score%':>6} {'sc_host ms':>10} {'n_sc':>4} | {'q ms':>6} | {'ws gpu/host us':>15}")
print(hdr)
groups = collections.defaultdict(list)
for f in sorted(fwd):
    r, a = fwd[f], acc[f]
    if not a.get("score_n"):
        continue
    F = r["gpu_us"] / 1e3
    idx, sc = a["indexer_extend_gpu"] / 1e3, a["score_gpu"] / 1e3
    row = dict(F=F, idx=idx, sc=sc, sch=a["score_host"] / 1e3, q=a["q_inputs_gpu"] / 1e3,
               ws=a["ws_build_gpu"], wsh=a["ws_build_host"])
    if a.get("q_wqb_n"):
        row.update({k: a[k + "_gpu"] / 1e3 for k in ("q_wqb", "q_rope_fq", "q_pack", "q_weights")})
    print(f"{f:>4} {r['tokens']:>6} {r['reqs']:>4} {r['max_seq']:>7} | {F:7.1f} | {idx:7.2f} {100 * idx / F:5.1f} | "
          f"{sc:8.2f} {100 * sc / F:6.1f} {row['sch']:10.2f} {int(a['score_n']):>4} | {row['q']:6.2f} | "
          f"{row['ws']:7.0f}/{row['wsh']:<7.0f}")
    groups[(r["tokens"], round(r["max_seq"], -3))].append(row)

print("\nmedian per (tokens, max_seq~1k):")
for (tok, ms), rows in sorted(groups.items()):
    m = {k: statistics.median(x[k] for x in rows) for k in rows[0]}
    print(f"  tok {tok:>6} seq~{ms:>7} n={len(rows):>2}: fwd {m['F']:7.1f} ms, indexer {m['idx']:6.2f} ms "
          f"({100 * m['idx'] / m['F']:4.1f}%), score {m['sc']:6.2f} ms ({100 * m['sc'] / m['F']:4.1f}%), "
          f"score host {m['sch']:5.2f} ms, ws {m['ws']:.0f}/{m['wsh']:.0f} us")
    if "q_wqb" in rows[0]:
        print("      q split (ms, sum over layers): " + ", ".join(
            f"{k} {m[k]:.2f}" for k in ("q_wqb", "q_rope_fq", "q_pack", "q_weights")))
