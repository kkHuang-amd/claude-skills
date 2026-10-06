"""Deliverable 3: kernel summary of one rank-0 vLLM torch trace.
usage: kernel_summary.py <trace.json.gz> [--extend]   -> markdown on stdout
GPU time sums over streams (kernels overlap across streams), so group pct is of summed kernel time;
wall = first->last GPU event; idle = wall not covered by any GPU kernel/memcpy/memset.
"""
import collections, gzip, json, re, sys

GROUPS = [
    ("sparse MLA attention", r"fmhaSm100|flashinfer_mixed_sparse_indices|mla"),
    ("indexer (logits + top-k)", r"mqa_logits|topk|mbtopk|gatherTopK|computeBlockDigitCounts|radixSort|_block_scores|indexer"),
    ("KV compressor / compressed-KV", r"Cache|compress|kv_insert|kv_store|paged_kv"),
    ("MoE (incl. routing)", r"^bmm_|moe::|routing|finalizeKernel|expert|fused_moe"),
    ("mHC (hyper-connections)", r"mhc|hc_prenorm|hc_post|hc_pre"),
    ("all-reduce/comm", r"allreduce|Allreduce|nccl|AllGather|all_gather|reduce_scatter|lamport|oneshot|twoshot"),
    ("engram", r"engram|Engram"),
    ("sampling/draft-specific", r"sample|gumbel|dflash|dspark|draft|rejection|argmax|softmax|logprob|penalt"),
    ("dense GEMM", r"gemm|Gemm|GEMM|nvjet|cublas|cutlass|splitK|sm100_xmma|ampere_|sm90_"),
    ("norm/rope/elementwise/quant", r"norm|rope|rotary|elementwise|act_and_mul|silu|quantiz|Quantiz|copy|Fill|fill|cat|index|scatter|gather|add|mul"),
]
CRE = [(g, re.compile(p)) for g, p in GROUPS]


def group(name):
    if "quantiz" in name.lower():  # flashinfer quant kernels are named kernel_cutlass_*; keep them out of GEMM
        return GROUPS[-1][0]
    for g, r in CRE:
        if r.search(name):
            return g
    return "other"


def union(iv):
    iv.sort(); tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if cs is not None else 0)


def main():
    path = sys.argv[1]; extend = "--extend" in sys.argv
    ev = json.load(gzip.open(path))["traceEvents"]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    steps = [e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_")]
    k = [e for e in gpu if e["cat"] == "kernel"]
    t0 = min(e["ts"] for e in gpu); t1 = max(e["ts"] + e["dur"] for e in gpu)
    wall = t1 - t0; busy = union([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
    tot = collections.Counter(); calls = collections.Counter(); gt = collections.Counter()
    for e in k:
        tot[e["name"]] += e["dur"]; calls[e["name"]] += 1; gt[group(e["name"])] += e["dur"]
    ksum = sum(tot.values()); ns = max(len(steps), 1)
    names = collections.Counter(s["name"] for s in steps).most_common(3)
    print(f"trace `{path.split('/')[-1]}`; steps (execute_* annotations): {len(steps)} {names}")
    kstreams = collections.Counter(e["args"].get("stream") for e in k)
    print(f"kernel streams: {len(kstreams)}; sum/busy {ksum / busy:.3f}x")
    if extend:
        print(f"wall GPU window {wall/1e3:.2f} ms, summed kernel time {ksum/1e3:.2f} ms, GPU idle {100*(1-busy/wall):.1f}%")
    else:
        print(f"wall GPU window {wall/1e3:.2f} ms -> **{wall/1e3/ns:.3f} ms/step** over {ns} steps; summed kernel time "
              f"{ksum/1e3/ns:.3f} ms/step (multi-stream overlap); GPU idle {100*(1-busy/wall):.1f}%")
    print("\n| group | total ms | " + ("" if extend else "ms/step | ") + "pct |")
    print("|---|---:|" + ("" if extend else "---:|") + "---:|")
    for g, v in sorted(gt.items(), key=lambda x: -x[1]):
        print(f"| {g} | {v/1e3:.2f} | " + ("" if extend else f"{v/1e3/ns:.3f} | ") + f"{100*v/ksum:.1f} |")
    print("\n```csv\nkernel_name,group,calls,total_us,us_per_step,pct")
    for n, v in tot.most_common(30):
        print(f"\"{n[:120]}\",{group(n)},{calls[n]},{v:.0f},{v/ns:.1f},{100*v/ksum:.2f}")
    print("```")
    oth = [(n, v) for n, v in tot.most_common() if group(n) == "other"][:8]
    if oth:
        print("\nlargest 'other': " + "; ".join(f"{n[:80]} {v/1e3:.2f}ms" for n, v in oth))


if __name__ == "__main__":
    main()
