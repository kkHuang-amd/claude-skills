# Session prompt — after the 2026-09-03 image swap (paste everything below the line)

---

Read `/workspace/claude-skills/NEW_WORKSPACE_PROMPT.txt` and adopt its rules for
this whole session (token discipline, capped tool output, batched calls, docs as
durable state). Then read, in ONE batched message and nothing else to start:

- `/workspace/claude-skills/agentx/IMAGE_SWAP_HANDOFF_20260903.md` — **in full**.
  It is short and it is the only thing standing between you and a destroyed
  source tree.
- `/workspace/claude-skills/agentx/ITL_GAP_FINDINGS.md` — the `CONTINUE HERE`
  block at the top, plus §13, §14 and §15. Skip §1–§10 unless you touch the
  ITL/TBO history.

Do NOT read `SKILL.md` (129 KB, describes a destroyed node). Do NOT read
`mori/` (690k tokens in one file). Do not reconstruct prior chat history.

`AGENTX_20260901.md` belongs to a **concurrent session**. Read its
`CONTINUE HERE` only if you touch the OOR/mem-frac track, and **never rewrite
it** — an earlier session's edits to it were silently overwritten. Put your own
results in `ITL_GAP_FINDINGS.md`.

## The one thing that will bite you first

`/sgl-workspace` is on the **container overlay** and did not survive the image
swap. `/workspace` (ext4) and `/shared_nfs` (xfs) did. Both source trees and all
uncommitted work were exported to `/workspace/handoff-20260903-image-swap/`
before the swap. Scripts, tools and every arm's artifacts are safe under
`/workspace/claude-skills/agentx/` and `/workspace/results/`.

## Task 1: rebuild the tree (do this before anything else)

The new image already carries the FP4 indexer (#37353, upstream `f8cbf000f4a5`),
so our old hand-applied integration commit is **obsolete — do not apply it**.
Two things are missing.

**a) sglang PR #37660** (the FP4 indexer OOR fix; still open, approved, not
merged). Fetch it rather than replaying our patch — ours was generated against a
hand-applied #37353 and its `indexer.py` differs by 149/123 lines, mostly base
drift.

```bash
cd /sgl-workspace/sglang
git fetch https://github.com/sgl-project/sglang.git refs/pull/37660/head:pr37660
git cherry-pick e5d8c6dc82594989cccdee7f71e741da220cde9c
```

`e5d8c6dc8259` sits directly on `f8cbf000f4a5`, the same base the new image has,
so it should apply cleanly. `fp4_indexer_hip.py` was byte-identical between our
version and the PR, so only `indexer.py` is interesting.

**Expect one deviation to be needed again.** Upstream calls
`self.flashinfer_topk_transform(...)`, which does not exist — **not in our old
tree and not on the PR branch either** (verified, 0 definitions). We substituted
`topk_transform_512_flashinfer_unfused(...)` with an inline comment. Check
whether the new image defines it; if not, substitute again and say so.

**b) the aiter changes**, all uncommitted, all FP4-related. Old HEAD was
`c16d44b93` (#4811) — compare first.

```bash
cd /sgl-workspace/aiter
git apply --3way /workspace/handoff-20260903-image-swap/aiter-tracked.patch
git apply --3way /workspace/handoff-20260903-image-swap/aiter-staged.patch
tar xzf /workspace/handoff-20260903-image-swap/aiter-untracked.tgz -C .
```

If aiter's HEAD moved, resolve `pa_mqa_logits_fp4_prefill.py` and
`pa_mqa_logits_fp4.py` first — those are the flydsl MQA-logits kernels the FP4
indexer actually calls. The CSVs are tuning tables and conflict harmlessly.
`jit/flydsl_cache/` was excluded on purpose, so the first FP4 arm pays JIT
compile time.

**Verify, then stop and report** before spending an arm:

```bash
cd /sgl-workspace/sglang
rg -c 'logits_rows_per_chunk' \
  python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
  python/sglang/srt/layers/attention/dsv4/indexer.py
python -c "import sglang.srt.layers.attention.dsv4.indexer, \
sglang.kernels.ops.attention.dsv4.fp4_indexer_hip; print('imports ok')"
```

## Task 2: one bridge arm, because every number on the board is now stale

The swap moved sglang, aiter, ROCm and torch all at once. §1's table is a record
of the **old image**. Until a bridge exists, **do not quote any cross-swap
delta** — not even the non-FP4 rows, which were still valid an hour ago.

Run `interval20_c128.sh` (or `dptbo_c128.sh`). Both have a pre-swap partner, so
either one pins how much the image alone moved. ~92 min: 3 min server start,
~28–37 min aiperf dataset config + warmup, 60 min measurement, ~5 min export.

```bash
mkdir -p /workspace/results/<arm>
nohup bash /workspace/claude-skills/agentx/<script>.sh \
  > /workspace/results/<arm>/run.log 2>&1 &
```

Pre-swap reference, `interval20-c128`: 29,130 tok/s/GPU, ITL p90 58.78 ms, TTFT
13.22 s, cache 94.2 %, `max_total_num_tokens` 6,979,584. A KV pool more than ~2 %
below that means the node was holding VRAM at startup and the arm is not
comparable — see the VRAM gate below.

## Task 3 (P0 once the tree is rebuilt): the fp8-path OOR fix

**This is the most urgent item and it is code, not an arm, so it costs no node
time.** PR #37660 bounds the **FP4** path only. The fp8 path still has the
unbounded `torch.empty(total_tokens, max_seq_len)` (the 13.7 GB transient) at
`dsv4/indexer.py:160` in `_aiter_fp8_paged_mqa_logits`.

Why now: `hicache-fp4-int20-c128` ran with free VRAM p10 **0.08 GB**, min
**0.01 GB** and **10 late Triton device loads** — the exact signature that killed
`fp4-dptbo-c64-reclaim0` (`HSA_STATUS_ERROR_OUT_OF_RESOURCES` on a code-object
load at 0.00 GiB). It survived on luck. Every arm from here is a coin flip.

Two parts, both upstream-shaped and worth offering back as a follow-up to #37660:

1. Mirror #37660's bounded-buffer + row-chunk pattern into
   `_aiter_fp8_paged_mqa_logits`.
2. Pre-load the Triton specialisations at engine init —
   `unified_kv_kernels/runtime.py:298,332`, where `BLOCK` has ≤ ~11 power-of-2
   values × `HAS_COMPRESS` × `compress_ratio` ∈ {0,4,128}. This is what the
   watchdog message (`srt/utils/triton_load_watch.py:115`) actually asks for.

Verification: `rg -c 'device-loaded after serving started' server.log` → 0, and
driver-visible free VRAM in `vram.csv` should stop sitting near zero.

## Task 4: the interval sweep — finish task 1 of the old session

`--prefill-decode-interval 15` at c128, **hicache OFF**. The box is bounded by
int 10 (ITL 78.39 ms, TTFT 8.50 s) and int 20 (57.98–58.78 ms, 13.22–13.77 s).
ATOM c128 is (61.5 ms, 10.9 s), **strictly inside it**, so a middle value can
satisfy both axes. If 15 lands ITL under and TTFT over, try 12.

**Pass criterion, scored as a PAIR:** ITL p90 ≤ 61.5 ms **and** TTFT avg
≤ 10.9 s **simultaneously**. That is the reportable result — "SGLang matches
ATOM on both axes at c128 once the prefill/decode split is tuned". Note the
reference numbers above are pre-swap; re-anchor them against the task-2 bridge
arm.

Leave hicache off: §15 measured it as a **no-op at c128** (CPU-tier hit 0.5 pp,
device pool 73 %, KV occupancy 0.27 — nothing to evict, nothing to recover), and
off keeps this a clean single-variable series.

Verify the override took: the launcher hardcodes
`--prefill-decode-interval 10` at `b200align_mtp.sh:235` and expands
`EXTRA_ARGS` later at `:372`, so **both 10 and 20 appear** and argparse keeps the
last. `server.log`'s `server_args` must show `'prefill_decode_interval': 15`.

## Then, in order

- **c256 at `HICACHE_RATIO` 5–6.** The only place the tier is provably the
  constraint: at c256 it hit **99.98 % full** with 18.5 pp of all reuse demoted
  to it, and we are −12.2 % on throughput vs ATOM's c256 (44,722 tok/s/chip).
  Host DRAM is 3,023 GB with 1,442.6 GB pinned at ratio 3, so 5–6 fits.
- Re-measure the FP4 curve. Mandatory now regardless.
- FP4-off replicate of the §15 arm, only if task 4 passes and clean attribution
  is then wanted.
- c224 to fill the curve. Low value.

## Settled — do not spend an arm re-litigating these

1. **TBO stays on.** TBO off is worse on ITL (83.93 vs 78.39 ms), TTFT (+19 %)
   and tok/s (−3.23 %). Both engines' TBO is prefill-only by construction.
2. **Do not profile a decode step.** A scheduler flag moved ITL by 20 ms, more
   than the whole gap. Decode is not slow; it was being interrupted.
3. **Do not re-run the chunk-size diagnostic.** Already measured: 16,384 → 8,192
   gives ITL −25 %, TTFT +147 %.
4. **The c192 collapse was prefill-capacity saturation, not eviction and not
   memory pressure** — and hicache fixed it outright (§13: +58 % tok/s, −85 %
   TTFT, queue p90 78 → 7, decode batch 15 → 23).
5. **hicache only pays where GPU KV pool occupancy is high** — 96 % at c192,
   100 % at c256, 73 % at c128 = nothing. Do not add it to low-concurrency arms.
6. **hicache and `--prefill-decode-interval` are orthogonal.** Interval 20's
   TTFT is policy-imposed deferral (`#queue-req` p90 = 9), not prefill demand,
   so cheaper prefill cannot pay it off. Do not combine them expecting addition.
7. **ATOM's `run 33074134043` is unreachable** — ROCm blocks all classic PATs.

## Launcher and metric traps that have already cost time

- **`KV_OFFLOADING=hicache` does not exist.** `benchmark_lib.sh:44-67` accepts
  only `none` and `dram`. Working combination: `KV_OFFLOADING=dram` +
  `KV_OFFLOAD_BACKEND=hicache` + positive integer `TOTAL_CPU_DRAM_GB`.
- **`KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'` is a third, undocumented
  requirement.** Without it `process_agentic_result.py:89` exits 1 **after a
  fully successful benchmark** and writes no result JSON. Do not re-run the
  benchmark — §13 has a one-command re-aggregate.
- **Window the miss rate.** `Σnew/Σ(new+cached)` over the whole `server.log` is
  contaminated by a ~37 min cold-cache warmup contributing about half the
  `Prefill batch` lines. Windowed reference: c96 5.59 %, c128 5.81/5.35 %,
  c160 5.96 %, c192 8.15 % → 5.01 % with hicache, c256 4.99 %.
- **Never report `gpu_cache_hit_rate` as "the cache hit rate"** on a hicache
  arm. At c256 the device tier read 0.766 while overall was flat at 0.951 — the
  hits demoted, they did not vanish. `arm_report.py` and `summary_table.py` are
  both fixed; anything quoted from an older `arm_report` run needs re-reading.
- **`ITL p90` is per-request TPOT**, p90 across requests — not the p90 of token
  gaps. It cannot show a bursty tail.
- **A dead server keeps answering `GET /metrics` with 200** and aiperf then waits
  forever. Verify an arm early, not at the end.

## Node traps

- **The node is shared.** Arm scripts wait for three consecutive idle checks and
  **refuse** rather than kill. Keep that. A blind kill preamble destroyed another
  session's c96 arm at 05:12:25 on 09-02, and the giveaway was misread as "my own
  leftover processes". Check `ps -eo pid,lstart,args` before touching anything.
- **The process check is not enough — copy the VRAM gate** from
  `hicache_fp4_int20_c128.sh`. Twice the node showed ~112–118 GB held evenly
  across all 8 GPUs with **zero visible processes** (a job in another container's
  PID namespace is invisible; its allocation is not). Starting an arm then
  silently shrinks the KV pool and breaks the memory matching the whole ATOM
  comparison rests on. The gate refuses rather than kills, and it works — the
  §15 arm got a 7,187,200-token pool, larger than its pre-swap partner's.
- **Never `pkill -f` a pattern matching your own command line.**
- **Do not edit a running bash script.** bash reads it by byte offset; editing
  mid-run corrupts execution.
- **`MEM_FRACTION_STATIC=0.90`, do not lower it.** ATOM runs
  `--gpu-memory-utilization 0.9`; 0.85 breaks the memory matching and invalidates
  every 0.90 arm on the board.

## What a valid result looks like

- Gates first: `errors=0`, `records_error_dropped=0`, `duration_s` 3500–3750,
  aiperf coverage ok. Report with
  `python3 /workspace/claude-skills/agentx/arm_report.py <arm> [<baseline>]`.
- ISL matched within 3 % between a pair; state which variables differ if more
  than one does.
- **Replicate spread on this node is 5.67 %** — a smaller throughput delta is
  null. Say "null", not "small improvement".
- After every arm: `python3 summary_table.py`, then re-sync it into
  `DATA_AND_ANALYSIS_20260902.md` §1, and keep `ITL_GAP_FINDINGS.md`'s
  `CONTINUE HERE` current.

Start with task 1, and report the rebuilt tree's state before running anything.
