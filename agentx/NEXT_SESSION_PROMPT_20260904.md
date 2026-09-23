# Session prompt — 2026-09-04 (paste everything below the line)

---

Read `/workspace/claude-skills/NEW_WORKSPACE_PROMPT.txt` and adopt its rules for
this whole session (token discipline, capped tool output, batched calls, docs as
durable state). Then read, in ONE batched message and nothing else to start:

- `/workspace/claude-skills/agentx/RESULTS_SUMMARY_20260904.md` — **in full**,
  including the CORRECTION block appended at the end. It is ~14 KB and is the
  single best picture of where the project stands.
- `/workspace/claude-skills/agentx/ITL_GAP_FINDINGS.md` — the `CONTINUE HERE`
  block at the top, plus §19, §20, §21. Skip §1–§18 unless you touch that
  history; they are superseded by the summary above.

Do NOT read `SKILL.md` (129 KB) unless you need an ATOM reference number. Do NOT
read `mori/` (690k tokens in one file). Do not reconstruct prior chat history.

## Verify these two things before you touch anything

**1. The tree. Two containers disagree, and one document is wrong about it.**

```bash
cd /sgl-workspace/sglang && git log -1 --format='%h %s'
```

Expected here: **`efaeb6f664`** ("pass Wc at runtime..."), on top of
`e485dc2436` and `1e41776161`, base `2641e427be`. aiter at `c16d44b93` with ~15
local files. If that is what you see, **the tree is complete — do not apply any
patch.** `RESULTS_SUMMARY` §5 claims the tree is back at pre-swap `33979a814b`;
that was true of a different container. Applying the patches on top of a tree
that already has them will conflict.

If the commits really are absent (fresh image), rebuild from
`/shared_nfs/kk/tree-backup/20260903-postswap/` — it has `REBUILD.md`,
`0001`–`0003`, `aiter-dirty.patch` and `aiter-untracked.tgz`. Note `git apply`
is all-or-nothing and the image may already carry some aiter changes; exclude
those files rather than fighting it (REBUILD.md explains).

**2. The node.** Shared, and another session is active in this workspace.

```bash
ps -eo pid,lstart,args | grep -E "[s]glang::|[a]iperf"
rocm-smi --showmeminfo vram | grep -oE 'Total Used Memory \(B\): [0-9]+' \
  | awk '{s+=$NF} END {printf "%.1f GB\n", s/1e9}'
```

Both should be empty / ~2.4 GB. Arm scripts gate on this and **refuse rather
than kill** — keep that. A blind kill preamble destroyed another session's c96
arm on 09-02.

## Where things stand in one paragraph

Against ATOM's published c256 point we are **−5.05 % on throughput** (42,462 vs
44,722 tok/s/chip) and we now **beat ATOM on ITL p90** (91.11 vs 97.6 ms) for
the first time at a matched concurrency. The entire remaining deficit is
**TTFT: 20.14 s vs 13.4 s.** It is policy-imposed deferral, not prefill demand
(miss rate unchanged at 4.96 %, queue p90 17), so the lever is
`--prefill-decode-interval`, not the cache.

## Pick one of these three. They are ordered by evidence, not by size.

**P1 — `HICACHE_RATIO` 5–6 at c256.** The only untested lever with a *measured*
reason to exist: the CPU tier is **exactly 100 % full** (22,047,744 of
22,047,744 tokens) with the GPU pool also at 100 %. Fusion's ~10 GB of device
headroom did nothing for it — device and host are independent resources. Host
DRAM is 3,023 GB with ~1,443 GB pinned at ratio 3, so 5–6 fits. Copy
`hicache_fp4_int20_fuse_conc.sh`, change `HICACHE_RATIO`, keep everything else.

**P2 — `--prefill-decode-interval` 15 at c128, hicache OFF.** The pair criterion
(ITL p90 ≤ 61.5 ms **and** TTFT ≤ 10.9 s *simultaneously*) is now missing by
**0.42 s of TTFT** with 3.98 ms of ITL margin in hand. That is the closest a
reportable "SGLang matches ATOM on both axes" has ever been. Verify the override
took: the launcher hardcodes interval 10 at `b200align_mtp.sh:235` and expands
`EXTRA_ARGS` later at `:372`, so **both 10 and 20 appear** and argparse keeps the
last — `server_args` must show `'prefill_decode_interval': 15`.

**P3 — an interval sweep at c256.** Never run. TTFT is the only axis still lost
there and the interval is the lever, so this is the direct attack on the
headline gap.

Lower value: re-measure the FP4 curve (every FP4 row predates both the scoring
change and the image swap); c224 to fill the curve.

## Things that will waste an arm if you do not know them

- **Validate ATOM before trusting the head-to-head.** 44,722 tok/s and 13.4 s at
  c256 are ATOM's *published* numbers and have **never been reproduced on this
  node** (`SKILL.md:1017`). "We beat ATOM on ITL" is against a published number,
  not a same-hardware run.
- **Cross-concurrency deltas are INVALID.** `arm_report.py` rejects them
  (`conc-and-trace-mix.md` 19.6); ISL moves ~10 % between c128/c192/c256. Quote
  the fusion curve as points (30,373 → 38,822 → 42,462 tok/s/GPU), never a %.
- **Replicate spread is 5.67 %** for throughput. A smaller delta is **null**,
  not a small improvement. There is still **no measured replicate spread for
  TTFT**, so do not score TTFT deltas against 5.67 %.
- **`MEM_FRACTION_STATIC=0.90`, do not lower it.** ATOM runs 0.9 and every arm
  on the board is 0.90. The 0.90→0.87→0.85 ladder in the fusion scripts exists
  for OOR only and **has never fired** — fusion freed ~10–14 GB.
- **Shared-experts fusion is a MEMORY result, not a throughput result.**
  +0.07 % tok/s at c128 is null. Enable it with `FUSE_SHARED_EXPERTS=1` (the
  launcher's own knob at `:188`), never via `EXTRA_SERVER_ARGS` — `EXTRA_ARGS`
  expands after `SHARED_EXPERTS_ARGS`, so an override puts both
  `--enforce-` and `--disable-shared-experts-fusion` on the command line.
  Its numerics are **unverified**: the shared experts are requantised FP8→FP4 at
  load time and the log says they "may differ slightly".
- **Late Triton loads are an ENGINE-WIDE property, ~510 per arm across a dozen
  kernels** (`alloc_extend_kernel` ~150, `_prefill_cta_info_kernel` ~65,
  `assign_req_to_token_pool` ~55). The two kernels we fixed are only ~64 of
  them. Per-kernel preloading does not converge. What keeps arms off the OOR
  cliff today is **fusion's headroom, not the preload** — say it that way.
- **Set `SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB=1000`** in any arm that reports
  a late-load count. The watchdog is silent above 1 GiB free, so a `0` on an arm
  with headroom means *not measured*. That confound already forced one
  retraction.
- **Window the miss rate.** Whole-log `Σnew/Σ(new+cached)` is contaminated by a
  ~37 min cold-cache warmup contributing about half the `Prefill batch` lines.
- **Never report `gpu_cache_hit_rate` as "the cache hit rate"** on a hicache
  arm — at c256 the device tier reads 0.822 while overall is 0.951; the hits
  demoted, they did not vanish.
- **`KV_OFFLOADING=hicache` does not exist.** Use `KV_OFFLOADING=dram` +
  `KV_OFFLOAD_BACKEND=hicache` + positive `TOTAL_CPU_DRAM_GB` + the undocumented
  `KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'`, without which
  `process_agentic_result.py:89` exits 1 **after a fully successful benchmark**.
- **A dead server keeps answering `GET /metrics` with 200** and aiperf then
  waits forever. Verify an arm early, not at the end.
- **Do not edit a running bash script.** bash reads it by byte offset.
- **The post-arm VRAM residue drains in ~13 min** (~119 GB, zero KFD processes,
  slow for the first two minutes then collapsing). Wait; do not escalate and
  never kill blindly. An earlier session wrongly recorded it as permanent.

## Settled — do not spend an arm re-litigating

TBO stays on. Do not profile a decode step. Do not re-run the chunk-size
diagnostic (16,384 → 8,192 = ITL −25 %, TTFT +147 %). hicache only pays where
GPU KV pool occupancy is high (73 % at c128 = nothing). hicache and the interval
are orthogonal. ATOM's `run 33074134043` is unreachable.

## Where things live

- Arm artifacts: `/workspace/results/<arm>/` (25 GB, `/workspace` has 512 GB
  free).
- **Logs, traces, scratch: `/shared_nfs/kk/`** (`logs/`, `traces/`, `tmp/`) —
  xfs, 53 TB, survives an image swap. `/sgl-workspace` does NOT.
- Durable rebuild kit: `/shared_nfs/kk/tree-backup/20260903-postswap/`.
- Scripts and docs: `/workspace/claude-skills/agentx/`. **Most of this directory
  shares one mtime (2026-09-04 02:05Z), i.e. a directory-level restore rather
  than per-file edits — another session can overwrite these docs.** Keep
  anything load-bearing in `/shared_nfs/kk/` as well.

## Reporting

Gates first: `errors=0`, `records_error_dropped=0`, `duration_s` 3500–3750,
aiperf coverage ok. Then
`python3 /workspace/claude-skills/agentx/arm_report.py <arm> [<baseline>]`.
State ISL match and the KV pool with every comparison — a pool more than ~2 %
below its partner means VRAM was held at startup and the arm is not comparable.
After every arm run `summary_table.py` and keep the `CONTINUE HERE` block in
`ITL_GAP_FINDINGS.md` current.
