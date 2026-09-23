#!/usr/bin/env python3
"""Where are the prefix-cache hits being served from, and are we losing them?

Two independent checks, because the obvious one cannot fire in this setup:

1. TIER SHARE (the literal question "how much is served from CPU DRAM").
   Requires a CPU/external tier to exist. With `KV_OFFLOADING=none`,
   `TOTAL_CPU_DRAM_GB=0` and no `--enable-hierarchical-cache` -- the state of
   every arm on this node, because PR #37353's rust pool-name variant was
   deliberately not applied -- `cpu_cache_hit_rate` and
   `external_cache_hit_rate` are null and `overall == gpu` exactly. The share is
   then 0 by construction and the check is vacuous. It is still worth printing:
   if it ever becomes non-zero, the configuration changed under us.

2. CEILING SHORTFALL (what actually degrades as concurrency climbs).
   With no CPU tier, a saturating GPU KV pool cannot demote hits -- it simply
   evicts them, and the workload silently reverts to cold prefill. The symptom
   is the achieved hit rate falling away from the trace's own theoretical
   maximum (`theoretical_prefix_cache_hit` in run.log), not appearing on another
   tier. This is the check that can actually stop a sweep.

Usage: cache_tier_gate.py <arm> [<arm> ...]
Exit 1 if any arm trips a threshold, so it can gate a driver script.
"""
import glob, json, os, re, sys

TIER_SHARE_MAX = 0.10      # >10 % of hits off-GPU
SHORTFALL_MAX = 0.10       # >10 % relative below the trace's own ceiling
GPU_HIT_FLOOR = 0.90       # absolute floor on the GPU-tier hit rate


def theoretical(d):
    lg = os.path.join(d, "run.log")
    if not os.path.exists(lg):
        return None
    last = None
    for ln in open(lg, errors="ignore"):
        m = re.search(r"theoretical_prefix_cache_hit=([\d.]+)%", ln)
        if m:
            last = float(m.group(1)) / 100.0
    return last


def check(arm):
    d = arm if arm.startswith("/") else "/workspace/results/" + arm
    c = sorted(glob.glob(os.path.join(d, "dsv4_fp4_sglang_*_c*.json")))
    if not c:
        print("%-22s NO RESULT JSON" % arm)
        return False
    cm = (json.load(open(c[0])).get("server_metrics", {}) or {}).get("cache") or {}
    gpu = cm.get("gpu_cache_hit_rate")
    cpu = cm.get("cpu_cache_hit_rate")
    ext = cm.get("external_cache_hit_rate")
    overall = cm.get("overall_cache_hit_rate") or gpu
    if overall in (None, 0):
        print("%-22s NO CACHE METRICS" % arm)
        return False

    off_gpu = (cpu or 0.0) + (ext or 0.0)
    # Cross-check that does not depend on the per-tier keys being populated.
    implied = max(0.0, overall - (gpu or 0.0))
    share = max(off_gpu, implied) / overall

    print("%-22s gpu=%.4f overall=%.4f cpu=%s ext=%s" % (
        arm, gpu or 0, overall, "null" if cpu is None else "%.4f" % cpu,
        "null" if ext is None else "%.4f" % ext))
    tripped = False
    if cpu is None and ext is None:
        print("   tier share    : n/a -- no CPU/external tier configured, so this "
              "cannot exceed 0 %. Check 2 is the one with teeth.")
    else:
        print("   tier share    : %.2f %% of hits off-GPU (limit %.0f %%)" % (
            share * 100, TIER_SHARE_MAX * 100))
        if share > TIER_SHARE_MAX:
            tripped = True

    # 3. ABSOLUTE FLOOR. The one that actually bites as concurrency climbs: with
    #    no CPU tier the pool cannot demote, so a saturating pool just evicts and
    #    the hit rate falls outright. c64 95.5 -> c96 94.5 -> c128 94.3 -> c160
    #    94.4 sat well clear of this; watch it from c192 up.
    print("   gpu floor     : %.2f %% (floor %.0f %%)" % (
        (gpu or 0) * 100, GPU_HIT_FLOOR * 100))
    if (gpu or 0) < GPU_HIT_FLOOR:
        tripped = True

    th = theoretical(d)
    if th:
        shortfall = (th - overall) / th
        print("   ceiling       : achieved %.2f %% vs trace max %.2f %% -> "
              "shortfall %.2f %% (limit %.0f %%)" % (
                  overall * 100, th * 100, shortfall * 100, SHORTFALL_MAX * 100))
        if shortfall > SHORTFALL_MAX:
            tripped = True
    else:
        print("   ceiling       : no theoretical_prefix_cache_hit in run.log")

    print("   VERDICT       : %s" % ("**STOP**" if tripped else "ok, may continue"))
    return tripped


if __name__ == "__main__":
    args = sys.argv[1:] or ["fp4-dptbo-c96"]
    sys.exit(1 if any(check(a) for a in args) else 0)
