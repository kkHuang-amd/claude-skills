#!/usr/bin/env python3
"""Emit the c64/c128/c256 summary table in the house format.

Rows are marked INVALID when aiperf refused to certify the run -- such a row's
latency columns are computed over the subset that finished and are biased
optimistic, so they are shown struck through rather than silently listed.
"""
import json, glob, os, re, sys

ROWS = [  # label, dir, chunk-per-rank
    ("DPA+TBO", "/workspace/results/b200align-tp8-c64-3600s", 8192),
    ("DPA+TBO", "/workspace/results/overnight/c64-chunk16384", 16384),
    ("DPA+TBO", "/workspace/results/overnight/c128-chunk8192", 8192),
    ("DPA+TBO", "/workspace/results/overnight/c128-chunk16384", 16384),
    ("DPA+TBO+topkv2", "/workspace/results/overnight/c128-chunk16384-topkv2", 16384),
    ("DPA+TBO newmain", "/workspace/results/c64-chunk16384-newmain", 16384),
    ("DPA+TBO", "/workspace/results/overnight/c256-chunk8192-2x", 8192),
    ("DPA+TBO", "/workspace/results/overnight/c256-chunk16384-2x", 16384),
]

def cov_failed(d):
    lg = os.path.join(d, "benchmark.log")
    if not os.path.exists(lg):
        return None
    for ln in open(lg, errors="ignore"):
        if "metric coverage below the required" in ln:
            m = re.search(r"TTFT=([\d.]+)%", ln)
            if m:
                return m.group(1)
    return None

def row(label, d, chunk):
    c = glob.glob(os.path.join(d, "dsv4_fp4_sglang_*_c*.json"))
    if not c:
        return None
    j = json.load(open(sorted(c)[0]))
    rm = j["request_metrics"]
    cm = j.get("server_metrics", {}).get("cache") or {}
    # OVERALL, not gpu_cache_hit_rate: once the GPU KV pool saturates, hits demote
    # to the CPU/HiCache DRAM tier and the device-tier number craters while true
    # cache effectiveness is unchanged (c256 reads gpu 0.66 / overall 0.94). Using
    # the device-tier number here is what produced the bogus "cache collapse" read
    # and the long-open TP8 0.689-vs-91.8% contradiction. See SKILL.md sec 15.
    cache = cm.get("overall_cache_hit_rate") or cm.get("gpu_cache_hit_rate")
    gpu_tier = cm.get("gpu_cache_hit_rate")
    kv = j.get("server_metrics", {}).get("kv_cache") or {}
    return dict(mode=label, conc=j.get("conc"), chunk=chunk,
                tps=rm["throughput"]["per_gpu"]["total_tput_tps"],
                intv=rm["latency"]["intvty"]["p90"],
                itl=rm["latency"]["itl"]["p90"] * 1000,
                ttft=rm["latency"]["ttft"]["mean"],
                ttft50=rm["latency"]["ttft"]["p50"],
                cache=cache, gpu_tier=gpu_tier, gpu_used=kv.get("gpu_usage_pct"),
                isl=rm["tokens"]["input"]["mean"],
                bad=cov_failed(d))

out = []
for label, d, ch in ROWS:
    r = row(label, d, ch)
    if r:
        out.append(r)
    else:
        out.append(dict(mode=label, conc=int(re.search(r"c(\d+)", d.split("/")[-1]).group(1)),
                        chunk=ch, pending=True))

print("| mode | conc | chunk/rank | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | TTFT p50 | cache hit | GPU-tier | GPU pool | ISL mean |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in out:
    if r.get("pending"):
        print("| %s | %d | %d | _running_ | | | | | | | | |" % (r["mode"], r["conc"], r["chunk"]))
        continue
    mark = " **[INVALID]**" if r["bad"] else ""
    print("| %s%s | %d | %d | %s | %.1f | %.1f ms | %.2f s | %.2f s | %.1f%% | %.1f%% | %.0f%% | %s |" % (
        r["mode"], mark, r["conc"], r["chunk"], "{:,.0f}".format(r["tps"]),
        r["intv"], r["itl"], r["ttft"], r["ttft50"],
        (r["cache"] or 0) * 100, (r["gpu_tier"] or 0) * 100,
        (r["gpu_used"] or 0) * 100, "{:,.0f}".format(r["isl"])))
if any(r["mode"].endswith("topkv2") for r in out):
    print("\n[topkv2] +3.75 % over c128-chunk16384 is INSIDE the 5.67 % replicate"
          " spread -- report as null, do not quote it as a win (sglang#36684).")
for r in out:
    if r.get("bad"):
        print("\n[INVALID] conc %d chunk %d: aiperf coverage TTFT=%s%% < 95%% -- run not"
              " certified; latency columns cover only requests that finished, so they are"
              " biased optimistic." % (r["conc"], r["chunk"], r["bad"]))
