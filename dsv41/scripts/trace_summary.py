#!/usr/bin/env python3
"""Summarize GPU kernel time in a torch profiler trace (.json.gz) by category. Usage: trace_summary.py <trace> [topN]"""
import gzip, json, re, sys
from collections import defaultdict
t = json.load(gzip.open(sys.argv[1])); top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ev = [e for e in t.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
CATS = [("moe", r"moe|fmoe|flydsl|expert|sorting|topk_softmax|grouped"), ("attn_sparse", r"sparse|mla|pa_|flash|attn|indexer|mqa|logits"),
        ("allreduce", r"allreduce|all_reduce|reduce_scatter|allgather|nccl|rccl|cross_device|quick_reduce"),
        ("gemm", r"gemm|matmul|Cijk|mfma|gemv|wo_a|wo_b|dot_scaled|hipblaslt|_mm_"), ("norm_rope_quant", r"norm|rope|quant|rotary|hc_|mhc"),
        ("memcpy", r"memcpy|memset|copy"), ("elementwise", r"elementwise|vectorized|index|cat|fill|reduce_kernel|arange")]
cat_t = defaultdict(float); name_t = defaultdict(float); name_n = defaultdict(int)
for e in ev:
    n = e.get("name", ""); d = e.get("dur", 0.0)
    c = next((k for k, p in CATS if re.search(p, n, re.I)), "other"); cat_t[c] += d
    short = re.sub(r"<.*", "", n)[:90]; name_t[short] += d; name_n[short] += 1
tot = sum(cat_t.values())
span = (max(e["ts"] + e["dur"] for e in ev) - min(e["ts"] for e in ev)) if ev else 0
print(f"kernels={len(ev)} gpu_busy={tot/1e3:.1f} ms span={span/1e3:.1f} ms (busy/span={tot/max(span,1):.0%})")
for c, v in sorted(cat_t.items(), key=lambda x: -x[1]): print(f"  {c:16s} {v/1e3:8.1f} ms {v/tot:6.1%}")
print(f"top {top} kernels:")
for n, v in sorted(name_t.items(), key=lambda x: -x[1])[:top]: print(f"  {v/1e3:8.1f} ms {v/tot:6.1%} x{name_n[n]:<5d} {n}")
