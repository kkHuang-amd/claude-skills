#!/usr/bin/env python3
"""Emit the c64/c128/c256 summary table in the house format.

Rows are marked INVALID when aiperf refused to certify the run -- such a row's
latency columns are computed over the subset that finished and are biased
optimistic, so they are shown struck through rather than silently listed.
"""
import json, glob, os, re, sys

# label, dir, chunk-per-rank.
# Reset 2026-09-01 for the rebuilt node: every arm the previous list pointed at
# was destroyed with /workspace/results and cannot be recovered. Add rows here
# as arms land; the previous node's numbers are NOT comparable (no EP, no MoRI).
ROWS = [
    # 2026-09-15 c256 pair. Same launcher/stack; only the MoE path and the memory
    # budget differ (megamoe needs mem-frac 0.85 to leave 16 GiB for the mori
    # heap, the DP arm keeps 0.92). MegaMoE wins every column here, which is the
    # opposite of the c128 pair below -- see SKILL.md.
    ("MegaMoE+EPLB EP8 DSPARK", "/workspace/results/megamoe-eplb-dp8-ep8-c256-mf085", 8192),
    ("DPA (no EP) DSPARK", "/workspace/results/dp8-noep-c256-d3600", 8192),
    # 2026-09-15 post-merge re-measure of the MegaMoE arm above, after
    # /sgl-workspace/sglang-MegaMoE took "Merge branch 'main'" (head ff7f522abd).
    # Identical launcher, env and aiter (ffa945f93 + its 2 uncommitted fixes);
    # the ONLY difference vs the row above is the merged sglang tree. Exists to
    # answer whether the merge cost throughput -- see dsv4/megamoe/
    # C256_REGRESSION_HANDOFF.md.
    ("MegaMoE+EPLB EP8 postmerge", "/workspace/results/megamoe-eplb-c256-postmerge", 8192),
    # 2026-09-15 evening: the postmerge row above, re-measured after aligning
    # three launcher values to the B200 sibling -- prefill-decode-interval 10->20
    # (CONC>=160 branch), load-balance-method round_robin->total_requests, router
    # policy consistent_hashing->cache_aware + --balance-abs-threshold 32. This is
    # a NEW operating point: it is the baseline for anything measured after that
    # edit, and it is NOT comparable to the 53,991 / 54,919 rows above.
    ("MegaMoE+EPLB EP8 B200-aligned", "/workspace/results/megamoe-eplb-c256-b200aligned", 8192),
    # 2026-09-16: same aligned config, re-measured on the rebuilt stack --
    # /sgl-workspace/sglang main 832ec39cc0 + merged PR #35619 (ff7f522abd), aiter
    # updated 4ad998328 -> ffa945f93 and fully rebuilt (126 .so, fresh FlyDSL cache),
    # fp4-prefill @cache fix reapplied. Pairs with the row above; see
    # dsv4/megamoe/ENV_SETUP_20260916.md.
    ("MegaMoE+EPLB EP8 aligned, rebuilt", "/workspace/results/megamoe-eplb-c256-20260916", 8192),
    # 2026-09-15: same launcher/stack as the row below, plus ENABLE_MEGAMOE=1 +
    # ENABLE_EPLB=1 at EP8. mem-frac 0.85 (not 0.92) because the 16 GiB mori
    # symmetric heap is charged outside that budget.
    ("MegaMoE+EPLB EP8 DSPARK", "/workspace/results/megamoe-eplb-dp8-ep8-c128-mf085", 8192),
    # 2026-09-16: the row above with the three B200-aligned launcher values. At
    # c128 the CONC>=160 branch does not fire, so this is PDI 24 with NO
    # --balance-abs-threshold -- a DIFFERENT aligned config from the c256 aligned
    # arm (PDI 20 + threshold). Compare each aligned arm to its own pre-align
    # row; the two aligned arms are not a concurrency sweep of one config.
    ("MegaMoE+EPLB EP8 B200-aligned", "/workspace/results/megamoe-eplb-c128-b200aligned", 8192),
    # 2026-09-17: the row above with ONE flag changed, --load-balance-method
    # total_requests -> total_tokens. A true single-variable A/B: cmd_diff.py
    # reports 46 flags with only that one differing, and the KV pool is identical
    # (max_total_num_tokens 12,077,312 both). Environment restored from the
    # reference arm's own launch log (mem-frac 0.85, MORI_SHMEM_HEAP_SIZE 16 GiB,
    # PYTHONPATH pinned to sglang-MegaMoE) -- the launcher's current defaults are
    # 0.65 / 40G and do NOT reproduce it.
    # Result: KV skew 2.08x -> 1.69x but the decode step did not move at matched
    # bs (0 to +2.4 %), so balancing is NOT an ITL lever here. TTFT 11.07 -> 9.61 s
    # (-13.2 %) with cache hit unchanged; +5.5 % throughput is INSIDE the 5.67 %
    # replicate spread and is a null. See exchange/FINDINGS.md.
    ("MegaMoE+EPLB EP8 aligned, total_tokens", "/workspace/results/megamoe-eplb-c128-b200aligned-totaltokens", 8192),
    # 2026-09-16: the c128 aligned row re-measured on the same rebuilt stack as the
    # c256 "rebuilt" row above.
    ("MegaMoE+EPLB EP8 aligned, rebuilt", "/workspace/results/megamoe-eplb-c128-20260916", 8192),
    # 2026-09-14: dsv4_fp4_mi355x_sglang_mtp.sh (DSPARK spec, no TBO, no EP) on
    # DeepSeek-V4-Pro-0813, sglang ab201bd1ba + rebuilt aiter ffa945f93 with the
    # fp4-prefill @cache fix restored. mem-frac 0.92, so it does not pair with
    # the 0.90 c128 rows below.
    ("DPA (no EP) DSPARK", "/workspace/results/dp8-noep-c128-d3600", 8192),
    ("DPA+TBO", "/workspace/results/dptbo-c64", 16384),
    ("DPA+EP8+MoRI mxfp8", "/workspace/results/mori-mxfp8-ep8-c64", 16384),
    ("DPA+EP8+MoRI mxfp8 +recvbound",
     "/workspace/results/mori-mxfp8-ep8-c64-recvbound", 16384),
    # c128 pair: mem-frac 0.90 (not the c64 rows' 0.85) and
    # HSA_NO_SCRATCH_RECLAIM=0, so these two compare to each other, not upward.
    ("DPA+TBO", "/workspace/results/dptbo-c128", 16384),
    ("DPA+TBO +FP4 indexer", "/workspace/results/fp4-dptbo-c128", 16384),
    # c96: same settings as the c128 FP4 arm, CONC=96 only. No c96 baseline yet,
    # so this extends the FP4 curve rather than forming a pair.
    ("DPA+TBO +FP4 indexer", "/workspace/results/fp4-dptbo-c96", 16384),
    # c64 + FP4 at mem-frac 0.90. Its partner (reclaim=0) aborted, so this row
    # has no matched baseline: the 0.85 dptbo-c64 row above differs in BOTH FP4
    # and mem-frac. Listed for completeness, not for a delta.
    ("DPA+TBO +FP4 indexer", "/workspace/results/fp4-dptbo-c160", 16384),
    # Past the knee: throughput peaks at c160 and c192 regresses 27 % with TTFT
    # 6x worse. c224 was queued behind a >=90 % cache gate, which passed (92.0 %)
    # and was the wrong indicator -- the failure mode is queueing, not cache.
    # It was killed unrun once c192 landed.
    ("DPA+TBO +FP4 indexer", "/workspace/results/fp4-dptbo-c192", 16384),
    ("DPA+TBO +FP4 indexer",
     "/workspace/results/fp4-dptbo-c64-reclaim1", 16384),
    # TBO off, the controlled partner for dptbo-c128. Worse on all three of ITL,
    # TTFT and tok/s, which is what settled the TBO question (sec 1).
    ("DPA, TBO off", "/workspace/results/dptbo-notbo-c128", 16384),
    # --prefill-decode-interval 10 -> 20, controlled partner for dptbo-c128.
    # Buys ITL p90 with TTFT; the operating point, not a deficit (CONTINUE HERE).
    ("DPA+TBO, interval 20", "/workspace/results/interval20-c128", 16384),
    # THE c192 FIX (sec 13). Single variable vs fp4-dptbo-c192: hicache CPU tier
    # at ratio 3.0. +58 % tok/s, -85 % TTFT, ITL flat -- and the best tok/s on
    # the board, above the old c160 "knee", which therefore was an artefact of
    # running with no CPU tier at all.
    ("DPA+TBO +FP4 +hicache", "/workspace/results/hicache-fp4-c192", 16384),
    # c256: CONC is the only difference from the c192 row above. Throughput keeps
    # rising (+7.9 %) but the CPU tier hits 100 % full and a fifth of all reuse
    # demotes to it, so ratio 3.0 -- not the engine -- is the constraint here.
    # Note `cache hit` is FLAT at 95.1 %: only the tier split moved (sec 14).
    ("DPA+TBO +FP4 +hicache", "/workspace/results/hicache-fp4-c256", 16384),
    # Both levers at once, first time: interval 20 + hicache + FP4 at c128.
    # Scored as a PAIR against ATOM c128 -- PASS needs ITL p90 <= 61.5 ms AND
    # TTFT avg <= 10.9 s simultaneously, which interval tuning alone never hit
    # (interval20-c128 got the ITL but overshot TTFT to 13.22 s).
    ("DPA+TBO +FP4 +hicache, interval 20",
     "/workspace/results/hicache-fp4-int20-c128", 16384),
    # BRIDGE ARM across the 2026-09-03 image swap: identical six settings to the
    # row above, new image (sglang e485dc2436, aiter + ROCm 7.2 + torch 2.9.1).
    # Everything ABOVE this line was measured on the old image. tok/s +2.11 % and
    # ITL -0.40 % vs its partner are both inside the 5.67 % replicate spread, so
    # the swap is null on both headline axes and the old rows stay quotable.
    ("DPA+TBO +FP4 +hicache, interval 20 [post-swap]",
     "/workspace/results/hicache-fp4-int20-c128-postswap", 16384),
    # Shared-experts fusion ON. NOT single-variable vs the row above: the Wc
    # runtime commit (efaeb6f664) landed between the two arms, so fusion and
    # that change moved together. tok/s +0.07 % and ITL -0.40 % are null;
    # the TTFT -6.95 % is NOT attributable to fusion (see findings sec 19).
    # Memory is the real result: weights 130.30 vs 133.75 GB/rank, KV pool
    # 7,412,480 vs 7,187,200, free VRAM med 15.45 vs 0.18 GB.
    ("DPA+TBO +FP4 +hicache, interval 20 +shared-experts fusion",
     "/workspace/results/hicache-fp4-int20-c128-fuse-mf090", 16384),
    # The fusion series continued to c192/c256, same six settings, CONC only.
    # These are the two rows that matter against ATOM, because interval 20 had
    # never been run with hicache above c128: ITL p90 falls to 75.57 / 91.11 ms
    # from the interval-10 hicache arms' 99.33 / 116.82 ms. At c256 that is
    # BELOW ATOM's 97.6 ms while throughput reaches 42,462 of ATOM's 44,722
    # (-5.05 %). TTFT is the axis still lost, 20.14 s vs 13.4 s. See sec 20.
    ("DPA+TBO +FP4 +hicache, interval 20 +shared-experts fusion",
     "/workspace/results/hicache-fp4-int20-c192-fuse-mf090", 16384),
    ("DPA+TBO +FP4 +hicache, interval 20 +shared-experts fusion",
     "/workspace/results/hicache-fp4-int20-c256-fuse-mf090", 16384),
    # 2026-09-04 -- A/B of the four decode-path PRs (#37423 / #37658 / #34624 /
    # #37580) on ONE tree: origin/main 8770c1db1f + the four merges, served by
    # PYTHONPATH from /shared_nfs/kk/tmp/combined. Config is byte-for-byte the
    # fusion series above; only the tree and the four env gates differ.
    #
    # READ THEM AS A PAIR, and only against each other. #37580 has no env var
    # (unconditional #ifdef USE_ROCM), so it is in BOTH rows -- the delta covers
    # three PRs, not four.
    #
    # VERDICT: null on EVERY timing metric, scored against the PER-METRIC floors
    # (headline 5.67 %, ITL p90 7.01 %, TTFT 26.3 %) -- tok/s +0.21 %,
    # ITL p90 -4.63 %, TTFT +12.98 %. Do NOT read the ITL gain as "decode got
    # cheaper": it is the shape these PRs predict, which is exactly why -4.63 %
    # against a 7.01 % floor must be called one noise draw. intvty p90 is not a
    # second witness, it is 1/p90(ITL).
    #
    # The ONE resolvable effect is the KV pool, 7,412,480 -> 7,664,896
    # (+252,416 tokens, +3.4 %) -- a deterministic allocation, no noise floor.
    # State this as a MEMORY result, like shared-experts fusion above.
    # The pool growth is a confound on the headline and is bounded to ~nothing:
    # occupancy FELL 66 % -> 54 % while capacity rose, and all three cache
    # numbers are identical, so the extra capacity was never a lever at c128.
    #
    # The OFF row doubles as the 62-commit base bump vs the c128 fusion row:
    # +1.53 % = null, and its KV pool is BIT-IDENTICAL (7,412,480), so that
    # comparison is memory-matched.
    #
    # Do not run a third arm on these flags. A 3600 s arm cannot resolve the
    # PRs' published +2.7 % / +2.95 % (needs ~8 runs/arm); profile at the kernel.
    ("combined 4 PRs on new main, gates OFF (paired baseline)",
     "/workspace/results/combined-prs-c128-off-mf090", 16384),
    ("combined 4 PRs on new main, gates ON (#37423+#37658+#34624)",
     "/workspace/results/combined-prs-c128-on-mf090", 16384),
    # Same pair at c192, where the pool is actually tight. The THREE c192 rows
    # (board / OFF / ON) decompose exactly, multiplicatively:
    #
    #            tok/s     TTFT      ITL p90   running p50   pool occ
    #   tree      -4.03 %  +46.74 %   -1.34 %   17 -> 16      96 -> 99 %
    #   gates     +0.84 %  +15.50 %   -7.77 %   16 -> 14      99 -> 85 %
    #   combined  -3.23 %  +69.49 %   -9.00 %   17 -> 14      96 -> 85 %
    #
    # THE HEADLINE IS THE TREE, NOT THE GATES. The 62 commits of new main cost
    # 46.7 % of TTFT and 4.0 % of throughput at c192 while being null-to-better
    # at c128 (+1.53 % tok/s, TTFT -5.69 %). It is a near-saturation penalty:
    # occupancy 96 -> 99 % on the same pool size. Read this before rebasing the
    # benchmark tree onto main.
    #
    # What the GATES buy at c192 is the memory result converting into cache
    # behaviour, which is what c128 could not show at 66 % occupancy:
    # occupancy 99 -> 85 %, GPU-tier hit 92.3 -> 94.2 %, CPU-tier dependence
    # 2.9 -> 1.0 pp. The +252,416-token pool gain is purely the gates -- the OFF
    # and board pools are BIT-IDENTICAL (7,380,992) at c192, as they were at c128.
    #
    # Throughput is null in every direction (+0.84 %, -4.03 %, -3.23 % vs a
    # 5.67 % floor). The gates' ITL p90 gain and TTFT cost reproduce their c128
    # SIGNS (-4.63 % / +12.98 %), which is the strongest claim available -- the
    # floors are all c64-derived and do NOT transfer to a c192 arm sitting next
    # to the documented capacity cliff.
    ("combined 4 PRs on new main, gates OFF (paired baseline)",
     "/workspace/results/combined-prs-c192-off-mf090", 16384),
    ("combined 4 PRs on new main, gates ON (#37423+#37658+#34624)",
     "/workspace/results/combined-prs-c192-on-mf090", 16384),
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

def mem_frac(d):
    """Read --mem-fraction-static off the arm's own server command.

    Not optional context: it decides which rows may be compared at all. The
    c64 arms ran 0.85 to match mori's out-of-budget symmetric heap and the
    c128/c96 arms ran the launcher's tuned 0.90, so a c64-vs-c128 delta on this
    table confounds the mode with the KV pool size.
    """
    cmd = os.path.join(d, "sglang_command.txt")
    if not os.path.exists(cmd):
        return None
    m = re.search(r"--mem-fraction-static\s+([\d.]+)", open(cmd, errors="ignore").read())
    return m.group(1) if m else None

def mem_facts(d):
    """KV pool, weight footprint, free-VRAM floor and late Triton loads.

    Added 2026-09-03 because sec 19's result was entirely in these numbers and
    none of them were on the table: shared-experts fusion moved tok/s by
    +0.07 % (null) while moving weights -3.45 GB/rank, the KV pool +3.13 % and
    the free-VRAM floor from 0.01 GB to 13.96 GB.

    `KV pool` is the memory-matching check. Rows may only be compared when it
    matches: it is what the ATOM comparison rests on, and a quietly shrunk pool
    (another container holding VRAM at startup) looks exactly like a real
    result. 7,187,200 is the c128 reference.

    `late loads` and `free VRAM p10` MUST be read together. triton_load_watch
    only reports a load when free VRAM is under
    SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB (default 1 GiB), so a 0 next to a
    p10 well above 1 GB means "not measured", NOT "none happened" -- which is
    exactly how sec 19's P0 gate reading came to be retracted. Arms from
    2026-09-03 on set the threshold to 1000 so the count is a real gate.
    """
    f = dict(kv=None, wt=None, late=None, vmin=None, vp10=None)
    log = os.path.join(d, "server.log")
    if os.path.exists(log):
        kv = wt = None
        late = 0
        with open(log, errors="ignore") as fh:
            for line in fh:
                if "device-loaded after serving started" in line:
                    late += 1
                if kv is None and "max_total_num_tokens=" in line:
                    m = re.search(r"max_total_num_tokens=(\d+)", line)
                    if m:
                        kv = int(m.group(1))
                if "mem usage=" in line:
                    m = re.search(r"mem usage=([\d.]+) GB", line)
                    if m:
                        v = float(m.group(1))
                        if wt is None or v > wt:
                            wt = v
        f.update(kv=kv, wt=wt, late=late)
    csv = os.path.join(d, "vram.csv")
    if os.path.exists(csv):
        vals = []
        with open(csv, errors="ignore") as fh:
            next(fh, None)
            for line in fh:
                p = line.rstrip("\n").split(",")
                if len(p) > 4:
                    try:
                        vals.append(float(p[4]))
                    except ValueError:
                        pass
        if vals:
            vals.sort()
            f.update(vmin=vals[0], vp10=vals[int(len(vals) * 0.1)])
    return f

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
    # Only worth a column when a CPU/external tier actually exists to demote to.
    # With hicache off both keys carry the same number, and printing it twice
    # invited the reader to treat one of them as a second, independent metric.
    gpu_tier = cm.get("gpu_cache_hit_rate")
    if gpu_tier is not None and cache is not None and abs(gpu_tier - cache) < 1e-9:
        gpu_tier = None
    # Split out where the hits are actually served from. Only meaningful once a
    # CPU tier exists: with hicache off this key is null and `overall == gpu`
    # exactly. Measured on hicache-fp4-c192 (sec 13): overall 95.1 % = GPU 93.7 %
    # + CPU 1.4 pp, so only ~45 % of the recovered hits are direct CPU reads and
    # the majority is the GPU tier hitting MORE often -- do not repeat the old
    # "a third of all reuse comes from CPU" framing off the previous node's c256.
    cpu_tier = cm.get("cpu_cache_hit_rate")
    kv = j.get("server_metrics", {}).get("kv_cache") or {}
    return dict(mode=label, conc=j.get("conc"), chunk=chunk, mf=mem_frac(d),
                tps=rm["throughput"]["per_gpu"]["total_tput_tps"],
                intv=rm["latency"]["intvty"]["p90"],
                itl=rm["latency"]["itl"]["p90"] * 1000,
                ttft=rm["latency"]["ttft"]["mean"],
                ttft50=rm["latency"]["ttft"]["p50"],
                cache=cache, gpu_tier=gpu_tier, cpu_tier=cpu_tier,
                gpu_used=kv.get("gpu_usage_pct"),
                isl=rm["tokens"]["input"]["mean"],
                bad=cov_failed(d), **mem_facts(d))

out = []
for label, d, ch in ROWS:
    try:
        r = row(label, d, ch)
    except (KeyError, TypeError) as e:
        # An arm that aborted mid-run still writes a result json, but a partial
        # one: the killed c96 attempt has no request_metrics.throughput.per_gpu.
        # One such row used to raise and take the whole table down, which is the
        # worst possible failure for a summary tool. Mark it and carry on.
        out.append(dict(mode=label, conc=int(re.search(r"c(\d+)", d.split("/")[-1]).group(1)),
                        chunk=ch, mf=mem_frac(d), partial=str(e)))
        continue
    if r:
        out.append(r)
    else:
        out.append(dict(mode=label, conc=int(re.search(r"c(\d+)", d.split("/")[-1]).group(1)),
                        chunk=ch, mf=mem_frac(d), pending=True))

# The tier columns are dropped entirely unless some arm actually ran a CPU tier.
# Before hicache landed, `overall == gpu` on every row and printing both invited
# the reader to treat one as a second, independent metric.
show_tier = any(r.get("gpu_tier") is not None or r.get("cpu_tier") is not None
                for r in out)
hdr = ["mode", "conc", "mem-frac", "chunk/rank", "tok/s/chip", "P90 intvty",
       "ITL p90", "TTFT avg", "TTFT p50", "cache hit"]
if show_tier:
    hdr += ["GPU-tier hit", "CPU-tier hit"]
hdr += ["GPU pool", "ISL mean", "KV pool", "weights GB", "free VRAM p10", "late loads"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for r in out:
    if r.get("pending") or r.get("partial"):
        note = "_running_" if r.get("pending") else "**[NO DATA]**"
        print("| %s | %d | %s | %d | %s |%s" % (
            r["mode"], r["conc"], r["mf"] or "?", r["chunk"], note,
            " |" * (len(hdr) - 5)))
        continue
    mark = " **[INVALID]**" if r["bad"] else ""
    cells = ["%s%s" % (r["mode"], mark), "%d" % r["conc"], r["mf"] or "?",
             "%d" % r["chunk"], "{:,.0f}".format(r["tps"]),
             "%.1f" % r["intv"], "%.1f ms" % r["itl"], "%.2f s" % r["ttft"],
             "%.2f s" % r["ttft50"], "%.1f%%" % ((r["cache"] or 0) * 100)]
    if show_tier:
        cells.append("%.1f%%" % (r["gpu_tier"] * 100) if r["gpu_tier"] is not None else "—")
        cells.append("%.1f pp" % (r["cpu_tier"] * 100) if r["cpu_tier"] is not None else "—")
    cells += ["%.0f%%" % ((r["gpu_used"] or 0) * 100), "{:,.0f}".format(r["isl"])]
    cells += ["{:,.0f}".format(r["kv"]) if r.get("kv") else "—",
              "%.2f" % r["wt"] if r.get("wt") else "—",
              "%.2f GB" % r["vp10"] if r.get("vp10") is not None else "—",
              # A NONZERO count is real evidence either way -- those loads did
              # happen. Only a ZERO is ambiguous: the watchdog is silent above
              # SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB (default 1 GiB), so a 0
              # on an arm that never dropped that low means "not measured".
              ("0 (n/m)" if (r["late"] == 0 and (r.get("vp10") or 0) >= 1.0)
               else "%d" % r["late"]) if r.get("late") is not None else "—"]
    print("| " + " | ".join(cells) + " |")
print("\n[legend] KV pool is the memory-matching check -- rows are only "
      "comparable when it matches (c128 reference 7,187,200). weights GB is the "
      "largest per-rank weight allocation. '0 (n/m)' under late loads means NOT "
      "MEASURED: triton_load_watch is silent above 1 GiB free (default "
      "SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB), so a zero next to a high free-VRAM "
      "p10 proves nothing. A nonzero count is always real.")
if any(r["mode"].endswith("topkv2") for r in out):
    print("\n[topkv2] +3.75 % over c128-chunk16384 is INSIDE the 5.67 % replicate"
          " spread -- report as null, do not quote it as a win (sglang#36684).")
for r in out:
    if r.get("partial"):
        print("\n[NO DATA] conc %d: the result json is incomplete (%s) -- the arm died"
              " mid-run, so it has no certifiable numbers." % (r["conc"], r["partial"]))
for r in out:
    if r.get("bad"):
        print("\n[INVALID] conc %d chunk %d: aiperf coverage TTFT=%s%% < 95%% -- run not"
              " certified; latency columns cover only requests that finished, so they are"
              " biased optimistic." % (r["conc"], r["chunk"], r["bad"]))
