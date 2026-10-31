#!/usr/bin/env python3
"""Per-step-type kernel breakdown by op class for one rank trace.
Usage: trace_breakdown.py <trace.json.gz> [--top N]
Steps come from gpu_user_annotation 'step[EXTEND|DRAFT|VERIFY ...]'."""
import bisect, gzip, json, re, sys
from collections import defaultdict

CLASSES = [
    ("comm", r"nccl|cross_device_reduce|allgather|all_gather|reduce_scatter|all_reduce"),
    ("moe", r"moe|k3_a8w4|fused_mx_quant_moe_sort"),
    ("sparse_mla", r"sparse_mla|mla_decode|opus_mla|flash_attn|mla_"),
    ("dsa_indexer", r"topk|mqa_logits|indexer|dcp_global_slots|dcp_topk"),
    ("kda", r"kda|delta_rule|causal_conv1d|layer_norm_gated|replayssm|chunk_"),
    ("attn_res", r"_agg_kernel|attn_res"),
    ("gemm", r"hgemm|gemm|Cijk_|_gemm"),
    ("norm_act", r"rmsnorm|layer_norm|situ|silu|gelu"),
    ("rope_cache", r"rope|kv_cache|set_kv|store_kv|concat_and_cache"),
    ("elementwise", r"at::native|add3|elementwise|copy|fill|index|cat"),
]
CRE = [(c, re.compile(p, re.I)) for c, p in CLASSES]


def klass(n):
    for c, r in CRE:
        if r.search(n):
            return c
    return "other"


path = sys.argv[1]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
ev = json.load(gzip.open(path))["traceEvents"]
steps = sorted(
    (e["ts"], e["ts"] + e["dur"], re.sub(r"\s.*", "", e["name"]).replace("step[", ""), e["name"])
    for e in ev
    if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step[")
)
starts = [s[0] for s in steps]
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]

by_type = defaultdict(lambda: {"n": 0, "span": 0.0, "busy": 0.0, "cls": defaultdict(float), "k": defaultdict(lambda: [0, 0.0])})
for s in steps:
    by_type[s[2]]["n"] += 1
    by_type[s[2]]["span"] += s[1] - s[0]
# Breakable prefill graphs run outside the EXTEND annotation (which only spans the
# graph prologue); kernels outside every step span are attributed to EXTEND_BODY.
outside = 0.0
by_type["EXTEND_BODY"]["n"] = max(by_type["EXTEND"]["n"], 1) if "EXTEND" in by_type else 1
for e in kern:
    i = bisect.bisect_right(starts, e["ts"]) - 1
    if i < 0 or e["ts"] > steps[i][1]:
        outside += e["dur"]; t = by_type["EXTEND_BODY"]
    else:
        t = by_type[steps[i][2]]
    t["busy"] += e["dur"]; c = klass(e["name"]); t["cls"][c] += e["dur"]
    a = t["k"][(c, e["name"])]; a[0] += 1; a[1] += e["dur"]

print(f"steps: {[(k, v['n']) for k, v in by_type.items()]}  kernel time outside steps: {outside/1e3:.1f} ms")
for st, t in by_type.items():
    n = t["n"]
    print(f"\n== {st}: n={n} mean GPU span {t['span']/n/1e3:.3f} ms, mean kernel busy {t['busy']/n/1e3:.3f} ms "
          f"(idle {100*(1-t['busy']/max(t['span'],1)):.1f}%)")
    for c, v in sorted(t["cls"].items(), key=lambda x: -x[1]):
        print(f"   {c:12s} {v/n/1e3:8.3f} ms/step {100*v/t['busy']:5.1f}%")
    print("   top kernels:")
    for (c, name), (cnt, v) in sorted(t["k"].items(), key=lambda x: -x[1][1])[:top]:
        print(f"     {v/n:8.1f} us/step {100*v/t['busy']:5.1f}% x{cnt/n:5.1f}/step [{c}] {name[:90]}")
