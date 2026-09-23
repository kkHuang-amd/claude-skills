# DeepSeek-V4-Pro / MI355X — measured data and analysis as of 2026-09-02

Consolidated handoff for a session that will integrate this with other sources.
Everything here is reproducible from `/workspace/results/<arm>/`. Companion
documents: `FP4_INDEXER_REPORT.md` (the FP4 integration write-up, whose §5 on the
OOR abort is **superseded** by §5 below) and `ITL_GAP_PROMPT.md` (the ATOM ITL
investigation brief, whose experiment ordering is **superseded** by §6 below).

Node: 8 × gfx950 (MI355X), TP8. Engine: SGLang `52e1c24744` + PR #37353 applied
to the working tree (uncommitted). Workload: InferenceX AgentX agentic trace
replay via aiperf 0.12.0 pinned at `754356e9`.

---

## 1. All certified arms

Every row below has `GATES PASS`: `errors=0`, `records_error_dropped=0`,
duration ≈ 3628 s, aiperf coverage ok. Common to all: chunk 16,384 tokens per
rank (`--chunked-prefill-size 131072`, which SGLang divides by `dp_size`), MTP
(EAGLE, 3 steps), fp8 KV cache, hicache off, DP attention.

| mode | conc | mem-frac | chunk/rank | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | TTFT p50 | cache hit | GPU-tier hit | CPU-tier hit | GPU pool | ISL mean | KV pool | weights GB | free VRAM p10 | late loads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DPA+TBO | 64 | 0.85 | 16384 | 20,730 | 21.4 | 46.8 ms | 4.62 s | 3.13 s | 95.5% | — | — | 58% | 116,596 | 6,129,920 | 133.75 | — | 0 |
| DPA+EP8+MoRI mxfp8 | 64 | 0.85 | 16384 | 8,710 | 7.8 | 128.9 ms | 12.91 s | 9.10 s | 95.4% | — | — | 73% | 114,463 | 5,980,672 | 134.53 | — | 101 |
| DPA+EP8+MoRI mxfp8 +recvbound | 64 | 0.85 | 16384 | 16,395 | 15.3 | 65.4 ms | 6.79 s | 4.33 s | 95.3% | — | — | 45% | 118,030 | 5,980,672 | 134.53 | — | 23 |
| DPA+TBO | 128 | 0.90 | 16384 | 27,895 | 12.8 | 78.4 ms | 8.50 s | 5.69 s | 94.2% | — | — | 71% | 99,954 | 6,979,584 | 133.75 | — | 2 |
| DPA+TBO +FP4 indexer | 128 | 0.90 | 16384 | 28,612 | 13.0 | 76.7 ms | 8.33 s | 5.68 s | 94.3% | — | — | 71% | 100,389 | 7,187,200 | 133.75 | — | 159 |
| DPA+TBO +FP4 indexer | 96 | 0.90 | 16384 | 25,138 | 15.6 | 64.1 ms | 6.98 s | 4.71 s | 94.5% | — | — | 57% | 102,746 | 7,203,072 | 133.75 | — | 35 |
| DPA+TBO +FP4 indexer | 160 | 0.90 | 16384 | 31,549 | 11.4 | 87.8 ms | 9.56 s | 6.22 s | 94.4% | — | — | 78% | 105,049 | 7,171,328 | 133.75 | — | 0 |
| DPA+TBO +FP4 indexer | 192 | 0.90 | 16384 | 22,994 | 9.9 | 101.1 ms | 59.71 s | 23.02 s | 92.0% | — | — | 77% | 111,977 | 7,155,456 | 133.75 | — | 0 |
| DPA+TBO +FP4 indexer | 64 | 0.90 | 16384 | 20,898 | 21.9 | 45.6 ms | 4.44 s | 3.07 s | 95.5% | — | — | 38% | 117,114 | 7,218,944 | 133.75 | — | 0 |
| DPA, TBO off | 128 | 0.90 | 16384 | 26,994 | 11.9 | 83.9 ms | 10.13 s | 6.38 s | 94.2% | — | — | 67% | 98,834 | 6,979,584 | 133.75 | 0.95 GB | 101 |
| DPA+TBO, interval 20 | 128 | 0.90 | 16384 | 29,130 | 17.0 | 58.8 ms | 13.22 s | 8.03 s | 94.2% | — | — | 65% | 101,762 | 6,979,584 | 133.75 | 2.64 GB | 6 |
| DPA+TBO +FP4 +hicache | 192 | 0.90 | 16384 | 36,414 | 10.1 | 99.3 ms | 9.11 s | 6.33 s | 95.1% | 93.7% | 1.4 pp | 96% | 110,515 | 7,155,456 | 133.75 | 4.74 GB | 0 (n/m) |
| DPA+TBO +FP4 +hicache | 256 | 0.90 | 16384 | 39,284 | 8.6 | 116.8 ms | 15.72 s | 9.64 s | 95.1% | 76.5% | 18.5 pp | 100% | 110,166 | 7,123,712 | 133.75 | 1.71 GB | 0 (n/m) |
| DPA+TBO +FP4 +hicache, interval 20 | 128 | 0.90 | 16384 | 29,726 | 17.2 | 58.0 ms | 13.77 s | 8.15 s | 94.8% | 94.3% | 0.5 pp | 73% | 102,377 | 7,187,200 | 133.75 | 0.08 GB | 10 |
| DPA+TBO +FP4 +hicache, interval 20 [post-swap] | 128 | 0.90 | 16384 | 30,352 | 17.3 | 57.8 ms | 12.16 s | 7.78 s | 94.9% | 94.5% | 0.4 pp | 75% | 103,000 | 7,187,200 | 133.75 | 0.04 GB | 74 |
| DPA+TBO +FP4 +hicache, interval 20 +shared-experts fusion | 128 | 0.90 | 16384 | 30,373 | 17.4 | 57.5 ms | 11.32 s | 7.95 s | 94.9% | 94.6% | 0.3 pp | 65% | 102,308 | 7,412,480 | 130.30 | 14.30 GB | 0 (n/m) |
| DPA+TBO +FP4 +hicache, interval 20 +shared-experts fusion | 192 | 0.90 | 16384 | 38,822 | 13.2 | 75.6 ms | 14.11 s | 8.88 s | 95.3% | 93.2% | 2.1 pp | 96% | 112,403 | 7,380,992 | 130.30 | 10.73 GB | 516 |
| DPA+TBO +FP4 +hicache, interval 20 +shared-experts fusion | 256 | 0.90 | 16384 | 42,462 | 11.0 | 91.1 ms | 20.14 s | 10.92 s | 95.1% | 82.2% | 12.9 pp | 100% | 112,351 | 7,349,248 | 130.30 | 10.38 GB | 503 |
[legend] KV pool is the memory-matching check -- rows are only comparable when it matches (c128 reference 7,187,200). weights GB is the largest per-rank weight allocation. '0 (n/m)' under late loads means NOT MEASURED: triton_load_watch is silent above 1 GiB free (default SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB), so a zero next to a high free-VRAM p10 proves nothing. A nonzero count is always real.

Regenerate with `python3 summary_table.py`. `mem-frac` is read per-arm from
`sglang_command.txt`, because it decides which rows may be compared.

**Updated 2026-09-03.** Three rows added: `dptbo-notbo-c128`, `interval20-c128`
and `hicache-fp4-c192`. Two headline changes to how this table should be read:

- **`hicache` is on in exactly one row, and it wins the table.** 36,414
  tok/s/chip at c192, above the old c160 peak, with TTFT 9.11 s instead of
  59.71 s. See `ITL_GAP_FINDINGS.md` §13 — it is a single-variable pair against
  the c192 row directly above it.
- **Therefore §2's "knee at c160" is an artefact of hicache being off**, and
  every row here except the hicache one was measured with **no CPU cache tier**.
  Do not quote this table as a concurrency curve for the engine; it is a curve
  for the engine *without* an L2 cache.
- The `GPU-tier` column is populated only when a CPU tier exists to demote to
  (93.7 % device against 95.1 % overall). On every other row the two numbers are
  identical by construction and the column is suppressed on purpose.

### Which comparisons are valid

- **Only one single-variable pair exists:** c128 at mem-frac 0.90, baseline
  27,894.8 vs FP4 28,611.7 = **+2.57 %**, with ISL matched to 0.43 % and cache
  hit flat. It is **inside the 5.67 % replicate spread**, so it does not
  establish an effect and still needs a replicate.
- The two c64 rows differ in **both** FP4 and mem-frac (0.85 vs 0.90);
  `arm_report.py` returns `VERDICT: INSIDE NOISE — report as null` at +0.81 %.
- The c96/c160/c192 FP4 arms have **no matched baseline**; they are a curve, not
  a comparison.
- The three MoRI rows are earlier EP/MoRI work, unrelated to FP4.

---

## 2. The FP4 concurrency curve: knee at c160

Throughput peaks at c160 and c192 regresses **27 %**, with TTFT 6.2× worse
(9.56 → 59.71 s) and a third fewer requests completed (8,651 → 5,914). Live
`queue=30r/76w` was observed during c192. This is over-saturation: requests
queue rather than being served.

**A cache-hit gate failed to detect it.** A ≥90 % GPU-tier hit floor was set to
decide whether to continue to c224. It **passed** at 92.0 %, c224 auto-launched,
and was killed once c192's numbers landed. Cache degraded only gently
(94.4 → 92.0 %; shortfall from the trace's own ceiling 1.80 → 4.38 %) while
throughput collapsed. **Past the knee, watch TTFT and completed requests, not
cache hit.** Gate tool: `cache_tier_gate.py`.

Caveat on the c160→c192 pair: c192's ISL is 6.6 % higher, past the 3 %
"different workload" flag, because heavy queueing changes which requests finish.
Direction is not in doubt given TTFT and completion counts.

---

## 3. `ITL p90` in this benchmark **is** TPOT

Verified in source, not assumed:

- aiperf `metrics/types/inter_token_latency_metric.py:22`:
  `Inter Token Latency = (Request Latency - Time to First Token) / (Output Sequence Length - 1)`
- InferenceX `utils/agentic/aggregation/request_metrics.py:190` pulls
  `inter_token_latency` **per record**, i.e. one value per request, then takes
  p90 across requests.
- `P90 intvty` = `1 / p90(ITL)` (`_interactivity_stats`).

So the table's `ITL p90` is "p90 across requests of each request's TPOT". It is
**not** the p90 of individual token gaps. Consequence: per-request averaging
already smooths individual stalls, so this metric cannot show a bursty tail, and
it cannot separate "the decode step is intrinsically slower" from "the decode
step waited for someone else's prefill". Both land in the same number.

---

## 4. Separating decode cost from prefill stalling — measured

Method (`decode_stall_split.py`): SGLang logs one `Decode batch` line every
`decode_log_interval` (default **40**) iterations carrying
`gen throughput (token/s)` for that window, so
`per_step_ms = running_req x accept_len / gen_throughput x 1000`. Each window is
then characterised by the `Prefill batch` lines that fall inside it.

**The classification must be cross-rank.** Under DP attention the ranks run in
lockstep (`prepare_mlp_sync_batch` all-gathers a common batch type), so a prefill
on *any* rank stalls *every* rank's decode. Classifying by a rank's own prefills
would have called 1,528 c192 windows clean instead of 119, and reported the
intrinsic step as 192 ms instead of 92.5 ms — inverting the conclusion to
"decode itself got slower".

### Do NOT use "% of windows containing a prefill" as evidence

An earlier revision reported 85–97 % of decode windows as "contaminated" and read
that as "almost no undisturbed decode". **That was wrong.** A log window is 40
iterations and `--prefill-decode-interval` is **10**, so 3–4 prefills fit in
every window *by construction*. The percentage is arithmetic on those two
numbers, not a statement about the engine.

A second problem: the genuinely prefill-free windows (238 at c160, 119 at c192)
are the moments with no prefill demand at all — i.e. light load, smaller running
batches — so they are a biased baseline that flatters the "intrinsic" figure.

### Dose-response is the load-bearing result

Per-step cost bucketed by how many prefills landed in the window. Every window
stays in the sample, so the light-load selection effect is diluted:

| prefills in window | c96 | c160 | c192 |
|---|---|---|---|
| 0 | 86.5 ms | 109.8 ms | 92.5 ms |
| 1 | 80.2 ms (-7 %) | 89.1 ms (-19 %) | 141.1 ms (+53 %) |
| 2 | 94.6 ms (+9 %) | 101.4 ms (-8 %) | 121.5 ms (+31 %) |
| **3** | 140.3 ms (**+62 %**) | 185.6 ms (**+69 %**) | 196.1 ms (**+112 %**) |
| >=4 | 124.5 ms (+44 %) | 186.1 ms (+70 %) | 203.9 ms (+120 %) |

**The relationship is not monotonic — it has a knee at 3.** One or two prefills
per window cost nothing measurable (c96 and c160 are even faster there); cost
jumps only at 3+.

That knee sits exactly where `--prefill-decode-interval 10` predicts it: with a
40-iteration window, the interval's own ceiling is 3–4 prefills. So **the decode
protection works** — at 1–2 prefills the decode gets enough uninterrupted runs
that per-step cost is unaffected. What hurts is saturating the quota: once
demand fills all 4 slots, per-step cost rises 62–120 %.

Caveats: `gen throughput` is SGLang's own in-window average, so these are mean
per-step figures and bound the stall's average, not its peak; tokens per
iteration is `running_req x accept_len` only while acceptance is steady; log
timestamps are 1 s resolution so a prefill on a window boundary can be
attributed to either side; the 0-prefill row is load-biased as described, so read
the percentages as direction, not magnitude. The unexplained speed-up at 1
prefill (-7 %, -19 %) is most likely that same bias.

Supporting fact: prefill is expensive because of the cached prefix. Per prefill
batch, `#cached-token` has p90 **416,640** and max **2,530,816** at c160 (max
3,877,120 at c192) — for only a few thousand new tokens.

---

## 5. The prefill budget is counted in the wrong unit

Attention work in a prefill is proportional to `new_tokens x context_length`,
where context = cached prefix + new. What SGLang bounds:

- `--chunked-prefill-size` (per rank after `// dp_size`) bounds **new tokens**.
- `SGLANG_PREFILL_TILE_BUDGET` (HIP-only, **off by default**) bounds query tiles
  via `estimate_prefill_extend_tile_metrics(extend_lens, block_m)` — again a
  function of extend lengths only.

Neither bounds the **cached/KV axis**. A batch of 1,792 new tokens over 137,728
cached tokens (a real c160 log line) consumes 11 % of the token budget while
doing `1792 x 139520 ~ 2.5e8` Q.K pairs — about the same work as a 16,384-token
uncached prefill that would consume 100 % of it. Observed `#cached-token` per
batch reaches 2.5–3.9 M.

ATOM has a knob on exactly that axis, `--attn-prefill-chunk-size` (default
16,384, **per-rank**), which splits the **cached prefix** into <=16,384-token
chunks and merges the partial attention results (`merge_attn_states`, online
softmax rescale). Total compute is unchanged; per `aiter_mla.py:1439` its purpose
is bounding the **decompressed k/v workspace**, and the chunk loop lives inside a
single forward so it yields nothing to the scheduler. **It is not an ITL
mechanism** — do not cite it as one.

### OOR abort — current state

Roughly half of the earlier hour-long arms died with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 0 MB`, at 2680 s and
2731 s of the measurement window. Ruled out: KV pool (usage 0.21–0.39 at abort),
torch OOM (never appears in any log), broken workload, corrupted tree, and
`HSA_NO_SCRATCH_RECLAIM` (a 2x2 gave a crossed pattern — each setting has one
survival and one abort, so the earlier "=0 is the fix" claim is **retracted**).

VRAM sampling was added from c96 onward (`vram_samples.txt`, 30 s cadence).
Driver-visible free VRAM is **flat**, with no creep: **1.0 GB** median at c96 but
**4.4 GB** at c160 — headroom does *not* shrink with concurrency, which weakens
the "0.90 leaves no room for HSA scratch, drop to 0.85" hypothesis. None of the
five sweep arms (c64...c192) aborted.

**Cause found — it is a lazily device-loaded Triton kernel, and it explains why
headroom size alone never predicted the abort.** The line immediately before the
abort in `fp4-dptbo-c64-reclaim0/server.log`:

```
[DP7 TP7] Triton kernel '_prefill_lengths_kernel' device-loaded after serving
started (free device mem: 0.00 GiB). Pre-load it during engine init to avoid CUDA OOM.
:0:rocdevice.cpp:3582: Callback: Queue ... HSA_STATUS_ERROR_OUT_OF_RESOURCES
```

So the failing allocation is a **code-object load** (`hipModuleLoad`), not a
tensor — which is why no torch OOM ever appears. SGLang has a watchdog for
exactly this: `srt/utils/triton_load_watch.py:115`. The kernels are
`_prefill_lengths_kernel` / `_build_prefill_indices_kernel`
(`kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py:298,332`), on the
unified-KV path this launcher selects with
`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`. They load mid-run because
`BLOCK=min(1024, next_power_of_2(max(win, Wc, 1)))`, `HAS_COMPRESS` and `Wc` are
**constexpr** specialisations and `Wc = page_idx.shape[1]` grows with context
length — one code object per combination, so every new power-of-2 bucket triggers
a load onto a device whose torch pool already holds everything.

Every arm walks this cliff; falling off is luck, which is exactly why the
`HSA_NO_SCRATCH_RECLAIM` 2×2 came out crossed and why none of the five sweep arms
died: `dptbo-c128` 2 loads at 0.17 GiB free (survived), `fp4-dptbo-c128` at
0.09 GiB (survived), `dptbo-notbo-c128` **101 loads**, several at 0.00 GiB
(survived), `fp4-dptbo-c64-reclaim0` 2 loads at 0.00 GiB (**died**).

It also adds the missing step to the "transient tensor cannot be the cause"
argument, which is otherwise right: **the transient never fails, it sets torch's
high-water mark.** The caching allocator does not return segments, so a 13.7 GB
transient permanently converts that much driver-visible memory into torch cache.
That is why free sits flat near zero rather than creeping.

**Acted on (2026-09-02):** upstream commit `b6e3728143` — "use a bounded prefill
logits buffer and process oversized batches in row chunks",
`SGLANG_DSV4_FP4_LOGITS_BUDGET_MB` default 2048 — has been **cherry-picked into
the working tree** as `33979a814b`, on top of `83310485e1` which commits the
by-hand #37353 integration. Other sessions' uncommitted files were left
untouched (19 entries, byte-unchanged); backup at
`/workspace/tree-backup-20260902-072044/`. One deviation was needed: upstream's
`self.flashinfer_topk_transform` does not exist in this tree, so the flashinfer
branch keeps `topk_transform_512_flashinfer_unfused`. Full record and the revert
path in `ITL_GAP_FINDINGS.md` §4b.

**Consequence: every FP4 number in §1 is now stale** — the FP4 scoring path
changed. The c96/c160/c192 curve and the +2.57 % pair must be re-measured on
`33979a814b`, and new arms should state the SHA. Verification on the next FP4
arm: driver-visible free should rise by roughly 11 GB, and
`device-loaded after serving started` should stop coinciding with
`free device mem: 0.00 GiB`. Note the fix is **FP4-only**; the fp8 path keeps the
unbounded `torch.empty(total_tokens, max_seq_len, float32)` at
`dsv4/indexer.py:160`, so mirroring it there is the outstanding piece.

---

## 6. ATOM comparison — what is verified, and the revised next experiment

ATOM reference (ROCm/ATOM PR #2068): DPA 64 -> 21,888 tok/s/chip, ITL p90
34.9 ms, TTFT 10.5 s; DPA 128 -> 30,709, 61.5 ms, 10.9 s; DPA 256 -> 44,722,
97.6 ms, 13.4 s. Cache hit 93.4–94.5 %.

Verified comparable:

- **Metric definitions match.** ATOM's `per_chip = (SigmaISL + SigmaOSL)/duration/num_gpus`
  reproduces our number exactly: `8033 x (99,954 + 871.6) / 3629.4 / 8 = 27,895`
  vs `arm_report.py`'s 27,894.8. `P90 intvty = 1/p90(ITL)` also matches.
- **Workload is identical** flag for flag (scenario, dataset, 393 entries, seed
  42, 3600 s, trajectory 0.25/0.75, warmup-per-lane 10, idle-gap-cap 300,
  slice-duration 1.0, `--use-server-token-count`), same aiperf fork and pin.
- **Server config matches** on fp8 KV, 16,384 prefill tokens per rank
  (SGLang `--chunked-prefill-size 131072` / dp8 <-> ATOM
  `--max-num-batched-tokens 16384`, which `atom/model_engine/scheduler.py:443`
  uses without dividing by `dp_size`), memory 0.90, `max-num-seqs` = 2 x conc,
  3 speculative steps, accept length ~2.4 vs ATOM's golden 2.49.
- **Both engines run the same prefill-scheduling mechanisms, at the same
  values.** SGLang has `--enable-prefill-delayer` (on in every arm) **and**
  `--prefill-decode-interval 10`; ATOM has `ATOM_ENABLE_PREFILL_DELAYER=1` and
  `ATOM_PREFILL_DECODE_INTERVAL=10`. The implementations correspond:
  SGLang's `_arm_prefill_decode_interval` / `_should_defer_prefill`
  (`scheduler.py:1243-1260`, DP-synchronised) is the same post-prefill decode
  protection as ATOM's `_decode_interval_remaining`
  (`prefill_delayer.py:346-363`). An earlier revision claimed SGLang lacked this
  window; that is **retracted** — it has it, with the same value.

Note the two flags point in opposite directions and should not be conflated:
`--prefill-decode-interval` is a *minimum spacing* that protects decode, while
`--prefill-delayer-max-delay-passes` (default 30) is a *cap on waiting* that
protects prefill from starvation ("cap the delay by max_delay_passes" —
`prefill_delayer.py:286-290`).

Two findings that reframe the gap:

1. **TBO is settled and it is not the gap.** Three things, in order of strength:
   - **Measured** (`dptbo-notbo-c128` vs `dptbo-c128`, single variable, both
     `GATES PASS`, ISL matched 1.12 %, cache flat): TBO off is **worse** — ITL p90
     83.93 vs 78.39 ms (+7.07 %), tok/s/GPU −3.23 % (inside noise), TTFT 10.13 vs
     8.50 s (+19 %). Turning TBO off moves ITL **away** from ATOM's 61.5 ms, so
     the gap widens from +27 % to +36 %. **Keep TBO on.**
   - **Both engines' TBO is prefill-only, so it was never a decode-side
     confounder.** ATOM: bare `--enable-tbo` is argparse `const=prefill` →
     `enable_tbo_decode=False`, and decode-TBO would drop MTP's
     `spec_decode_metadata` (`fixed_seq_len/dsv4_fp4_mi355x_atom_mtp.sh:44`).
     SGLang: `DeepseekV4ForCausalLM._can_run_tbo`
     (`models/deepseek_v4.py:2840-2876`) requires
     `global_forward_mode.is_extend_without_speculative()`, and
     `batch_overlap/operations_strategy.py:182` **raises** `NotImplementedError`
     ("DeepseekV4 TBO only supports prefill (EXTEND)") otherwise — our arms ran a
     full hour without it.
   - The PR's claim that its table predates `--enable-tbo` is now **doubtful**,
     because InferenceX PR #2778 (merged 2026-08-31T18:56Z) added the agentic DP
     band *with* `--enable-tbo` and removed the `DP_ATTENTION=false` guard, so the
     cited reference run (`run 33074134043`) could not have produced DP cells
     without it. Unresolvable from here (ROCm CI blocks all classic PATs, HTTP
     403, two tokens tried). It no longer matters: since both sides are
     prefill-only and TBO-off is worse for us, either reading leaves the same
     next step.

   Detail and the trap that nearly inverted this: `ITL_GAP_FINDINGS.md` §2.
   `SGLANG_TBO_DEBUG=1` shows TARGET_VERIFY batches being *prepared* at capture
   time; that is model-agnostic batch prep DSV4 then declines to use, **not**
   evidence of decode TBO.
2. **The mechanism is present and working, but its quota is too permissive under
   saturation.** Sec 4 shows cost is flat at 1–2 prefills per 40-iteration window and
   jumps 62–120 % at 3+, which is exactly the ceiling `interval=10` allows. With
   400 k+ cached tokens per prefill on this workload, four prefills per window is
   enough to double per-step decode cost even at the mandated spacing.

**Revised experiment order** (supersedes `ITL_GAP_PROMPT.md`):

1. ~~Raise `--prefill-decode-interval` above 10~~ **DONE 2026-09-02, and it
   worked — this was the answer.** `interval20-c128` vs `dptbo-c128`, single
   variable, both `GATES PASS`, ISL matched 1.81 %, cache 0.942 both:

   | metric | interval 20 | interval 10 | ATOM c128 |
   |---|---|---|---|
   | **ITL p90** | **58.78 ms** | 78.39 ms | 61.5 ms |
   | tok/s/GPU | 29,130.5 | 27,894.8 | 30,709 |
   | TTFT avg | 13.22 s | 8.50 s | 10.9 s |
   | intvty p90 | 17.01 | 12.76 | 16.3 |

   **ITL p90 fell 25 % and now beats ATOM's 61.5 ms**, while throughput moved the
   same direction (+4.43 %, inside the 5.67 % noise band but corroborated by
   OSL +4.5 % and succ +2.6 %). The cost is TTFT, +55.5 %, which overshoots
   ATOM's 10.9 s. So both pre-registered criteria were tested: ITL passed, TTFT
   failed, and **the optimum lies between 10 and 20**.

   Two conclusions follow. **The ITL gap was an operating-point choice, not an
   execution deficit** — one scheduler flag moved ITL by 20 ms, more than the
   entire gap, so *profiling a decode step is no longer indicated*. And the §10
   trade-off framing is confirmed end-to-end: the two engines sit on one
   prefill-versus-decode curve, and we were simply further toward the prefill end.

   **Next:** `--prefill-decode-interval 15` at c128, pass criterion ITL ≤ 61.5 ms
   **and** TTFT ≤ 10.9 s simultaneously. Script: copy `interval20_c128.sh`.
2. Only if that is neutral, revisit `SGLANG_PREFILL_TILE_BUDGET` — noting it
   bounds query tiles, not the cached axis, so it is a weaker match to the
   observation.
3. ~~TBO on/off~~ **done, see finding 1** — TBO off is worse on every metric that
   matters. No further TBO arm is needed.

**But read §9 before running experiment 1.** The dose-response knee and the c192
collapse turn out to be two different problems, and raising the interval helps
one while hurting the other. §9 also reframes the ITL gap as an operating-point
choice rather than a deficit, which changes what "closing it" means.

## 9. The c192 collapse is prefill-capacity saturation — NOT cache eviction

**Corrected 2026-09-02 after review.** An earlier revision of this section called
it a "cache-working-set failure" and put a 70 % rise in prefill work per request
at the centre. The objection that killed it: cache hit only fell 2.4 pp
(94.4 -> 92.0 %) and the pool was never full, so eviction cannot be doing the
work. Both halves are right, and re-deriving the numbers with a clean denominator
shows the 70 % was mostly an artefact.

| arm | new tok total | new-seq | completed | **new tok / sequence** | new tok / completed | seq / completed | KV usage median | KV p90 |
|---|---|---|---|---|---|---|---|---|
| c96 | 71.8 M | 11,389 | 7,043 | 6,302 | 10,191 | 1.62 | 0.20 | 0.34 |
| c128 (no TBO) | 86.6 M | 13,459 | 7,862 | 6,433 | 11,013 | 1.71 | 0.31 | 0.45 |
| c160 | 106.4 M | 15,794 | 8,651 | 6,739 | 12,304 | 1.83 | 0.40 | 0.54 |
| **c192** | 123.7 M | 15,000 | 5,914 | **8,247** | 20,918 | **2.54** | **0.36** | 0.54 |

Two corrections fall out:

1. **Per admitted sequence, prefill work rose 22 %, not 70 %** (6,739 -> 8,247).
   The rest of the "70 %" was the denominator: sequences per completed request
   went 1.83 -> 2.54, because at c192 far more admitted sequences never became a
   completed aiperf record inside the window. Note the ratio is already 1.62 at
   c96, so part of it is structural, not a load effect.
2. **There is no memory pressure and therefore no capacity-driven eviction.**
   `full token usage` median is **0.36** at c192, *lower* than c160's 0.40, with
   p90 flat at 0.54. KV occupancy went **down**. A working-set story predicts the
   opposite, so it is falsified by our own data.

A 22 % rise per sequence is fully accounted for by the two modest things already
in the table — ISL +6.6 % (105,049 -> 111,977) and hit -2.4 pp — with no eviction
required.

### What the data does support

**c160 is at the prefill capacity ceiling; c192 offers ~20 % more load while each
sequence costs ~22 % more prefill work. At a saturated resource that is enough
for the queue to diverge.** The evidence, all direct:

- The shared resource is **step time**, and c192 reallocates it toward prefill:
  prefill throughput per rank rises 3,696 -> 4,296 tok/s (+16 %) while decode
  throughput per rank falls 271 -> 191 (-30 %). Prefill is not starved; decode is.
  Even with that extra prefill throughput, admissions could not match the offered
  load once per-sequence cost rose 22 %, so the queue still diverged.
- Careful with one number: `#new-token` median is **exactly 16,384** — the full
  per-rank chunk budget — in *every* arm including c96. That is about batch
  *composition* (one long request fills the budget), **not** evidence of a
  capacity ceiling. Do not cite it as "prefill is saturated"; the prefill
  throughput series above is the honest measure, and it is still rising at c192.
- Decode moves the opposite way to what saturation would predict:

| | c160 | c192 |
|---|---|---|
| decode `#running-req` median | 20 | **15** |
| `#queue-req` median / p90 / max | 1 / 5 / 39 | 6 / 45 / 87 |
| gen throughput per rank, median | 271 tok/s | **191 tok/s** |
| aiperf waiting queue, median | 6 | **75** |

The decode batch is **smaller** at c192. Requests are not in decode because they
are still waiting to be prefilled, and a smaller decode batch is less efficient
per step — which is where the 27 % throughput regression comes from, not from
decode getting intrinsically slower. Falling KV occupancy (0.40 -> 0.36) is the
same fact seen from the memory side: fewer requests are resident in decode
holding KV. TTFT is then simply queue wait: p50 23 s, mean 59.7 s, and the 2.6x
mean/median skew is the signature of a queue, not of slow service.

### Root cause: a cache *hit* does not make prefill cheap, and the prefix grew 60 %

The obvious objection to any prefill-side explanation is: with 92 % prefix hit
and a KV pool only 36 % full, prefill should be reading its context out of cache.
It **is**. The hit saves recomputing the prefix's KV; it does **not** save the new
tokens from attending over that prefix, and prefill attention cost is
`new_tokens × (cached + new)` — the axis §5 shows nothing bounds.

Token-weighted prefix hit computed from the logs (`Σcached / (Σcached + Σnew)`),
with the length of the prefix each prefill actually attends over:

| arm | Σnew | Σcached | token-weighted hit | batches with cached=0 | non-zero cached, median | p90 |
|---|---|---|---|---|---|---|
| c96 | 71.8 M | 857.7 M | 92.3 % | 50 % | 143,488 | 579,840 |
| c128 | 86.6 M | 955.0 M | 91.7 % | 57 % | 175,616 | 688,384 |
| c160 | 106.4 M | 1,131.2 M | 91.4 % | 59 % | 189,824 | 731,136 |
| **c192** | 123.7 M | 934.0 M | **88.3 %** | **78 %** | **303,360** | **1,127,424** |

The hit barely moves (91.4 → 88.3 %), exactly as the objection says. **But
prefill work is proportional to the MISS rate, and at a 90 % hit "3 pp" is a
36 % change in the quantity that matters.** The exact decomposition, all sums
straight from the `Prefill batch` lines:

| | c160 | c192 | change |
|---|---|---|---|
| Σ new tokens | 106.4 M | 123.7 M | **+16.2 %** |
| Σ context (new + cached) | 1,237.6 M | 1,057.7 M | −14.5 % |
| **miss rate** = Σnew / Σcontext | **8.60 %** | **11.70 %** | **+36.0 %** |
| context per sequence | 78,359 | 70,517 | −10.0 % |
| new tokens per sequence | 6,739 | 8,247 | +22.4 % |
| **work per batch** (new × context) | 1.653e9 | 1.668e9 | **+0.9 %** |

It closes exactly: `70,517 × 11.70 % = 8,250` new tokens per sequence. The whole
+22.4 % is the miss rate's +36 %, partly offset by a 10 % *shorter* average
context.

**Two claims from an earlier revision of this subsection are hereby withdrawn.**

1. "The prefix per prefill grew 60 %." That was the median over *non-zero-cached*
   batches only, and the share of zero-cached batches jumped 59 → 78 %. The
   **average** context per batch *fell* 14.5 %. The distribution became more
   bimodal; the mean went the other way.
2. "Per-prefill time rose 1.01 → 1.38 s." Per-batch **work is flat (+0.9 %)**, so
   per-batch time should be flat too. The 1.38 s came from combining §4's
   0-prefill bucket with window arithmetic, and §4 warns that bucket is
   load-biased (0-prefill windows are light-load windows with smaller decode
   batches), which understates the c192 pure-decode baseline and inflates the
   inferred `T_prefill`. **What survives is the level, not the change:** a prefill
   batch costs order ~1 s against a decode step's ~0.1 s, so a prefill is a ~10×
   decode step, and under DP attention it stalls *every* rank
   (`prepare_mlp_sync_batch` all-gathers a common batch type).

### The mechanism, stated only as far as the data supports

- Individual prefills did **not** get bigger or slower at c192 (work/batch +0.9 %).
- What changed is **how much prefill the workload demands**: +16.2 % in total,
  +22.4 % per sequence, all of it from the miss rate.
- Prefill and decode compete for the same step time, and c160 was already at the
  knee. A 16 % rise in required prefill work with no spare capacity pushes
  admission below the offered load, so the queue diverges and TTFT becomes queue
  wait. Decode then gets both fewer steps (20,730 → 18,070 iterations/rank) and a
  smaller batch (20 → 15), which is the −30 % in gen throughput per rank.
- The pure-decode step is *faster* at c192 (92.5 vs 109.8 ms) because the batch is
  smaller. **Decode did not get slower; it got rarer.**

So the open question narrows to one thing: **why did the miss rate rise 36 %?**
That is precisely what the hicache arm tests — see the hicache bullet below,
where a c256 run shows a third of all reuse being served from the CPU tier, i.e.
prefixes that were absent from the GPU tier when their turn arrived.

### Consequences for what to try

- **Do not run the §6 interval sweep at c192.** Raising
  `--prefill-decode-interval` slows prefill admission, and at c192 prefill
  admission *is* the bottleneck. Run it at c128 or c160.
- **hicache / KV offload IS the primary lever. Both of my earlier positions on it
  were wrong and this one has evidence.** I first proposed it on an eviction
  theory, then retracted it because instantaneous `full token usage` is 0.36 and
  the hit fell only ~3 pp. That retraction used the wrong evidence: *instantaneous*
  occupancy cannot disprove eviction across a session's idle gap (the trace caps
  idle at 300 s), because other sessions' allocations churn through the pool while
  a session waits.

  The right evidence is the **CPU-tier hit share**, and a c256 run on the previous
  node supplies it: cache hit 94.2–94.4 % with the **GPU tier at only
  66.1–66.3 %** and total tier coverage 100 %. So roughly **one third of all
  prefix reuse was served from the CPU tier** — prefixes that were *not* in GPU
  memory when needed. With hicache off, every one of those tokens must be
  recomputed, which is exactly the `new_tokens` term in the cost model above.

  | node | conc | chunk | tok/s/chip | ITL p90 | TTFT avg | TTFT p50 | hit | GPU tier |
  |---|---|---|---|---|---|---|---|---|
  | previous, **hicache on** | 256 | 8,192 | 27,172 | **109.4 ms** | 94.13 s | 77.58 s | 94.2 % | 66.1 % |
  | previous, **hicache on** | 256 | 16,384 | 30,745 | 145.1 ms | **38.15 s** | 21.50 s | 94.4 % | 66.3 % |
  | this node, hicache off | 192 | 16,384 | 22,994 | 101.1 ms | 59.71 s | 23.02 s | 92.0 % | 77 % |

  c256 **with** hicache beats our c192 **without** it on both throughput
  (30,745 vs 22,994) and TTFT (38.15 vs 59.71 s) at a 33 % higher concurrency.
  Different node, so not a controlled pair — which is exactly why the next arm
  should be **c192 with `KV_OFFLOADING=hicache` on this node**.
- **The chunk diagnostic has already been run, and it confirms the stall model.**
  The c256 pair above differs only in `CHUNK_PER_RANK`. Halving it 16,384 → 8,192
  moves **ITL p90 145.1 → 109.4 ms (−25 %)** and TTFT **38.15 → 94.13 s
  (+147 %)**, throughput −12 %. That is the predicted trade-off measured
  end-to-end: shorter prefill chunks mean shorter stalls (better ITL) and lower
  prefill throughput (worse TTFT). So do **not** spend an arm re-running it here,
  and note that "16,384 is better than 8,192" is true on throughput and TTFT
  while false on ITL — which of the two matters is a choice, not a fact.
- **The mechanism-matched lever is the length of a single prefill stall, not how
  often prefills run.** `--prefill-decode-interval` changes the frequency; it
  cannot shorten a 1.4 s batch. Lowering `CHUNK_PER_RANK` below 16,384 does — it
  trades the same total prefill work into more, shorter stalls, so decode
  interleaves more finely. An earlier revision dismissed this knob for breaking
  the ATOM comparison, which is still true of the *headline* comparison but makes
  it the right **diagnostic**: if halving the chunk halves the stall and recovers
  ITL, the cached-axis cost is confirmed as the cause.
- `SGLANG_PREFILL_TILE_BUDGET` (HIP-only, off by default) is the other candidate,
  weaker because it bounds query tiles from extend lengths only (§5) — but it
  does cap work per prefill pass, which is the right *kind* of bound.
- c192 is past the knee, so the honest reporting position is that **c160 is this
  configuration's operating limit**, and ATOM's c256 point is not reachable until
  the per-prefill stall is bounded on the context axis.
## 10. The ATOM "gap" looks like a different operating point, not a deficit

Putting both engines' TTFT and ITL side by side, which neither document has done:

| conc | ATOM TTFT | ours | ATOM ITL p90 | ours |
|---|---|---|---|---|
| 64 | 10.5 s | **4.62 s** | 34.9 ms | 46.8 ms |
| 128 | 10.9 s | **8.50 s** | 61.5 ms | 78.4 ms |
| 160 | — | 9.56 s | — | 87.8 ms |
| 192 | — | 59.71 s | — | 101.1 ms |
| 256 | 13.4 s | not run | 97.6 ms | — |

**ATOM's TTFT is nearly flat across a 4× concurrency range (10.5 → 13.4 s) and is
*worse than ours* at every concurrency we have both.** We are 2.4–5.9 s better on
TTFT and 17–12 ms worse on ITL. That is not a uniform deficit — it is the shape
of two engines sitting at different points on the **same prefill-versus-decode
trade-off**, with ATOM biased hard toward protecting decode.

This turns the investigation into a falsifiable statement: **we have 2.4 s of
TTFT headroom at c128 and 5.9 s at c64 before we are worse than ATOM. Spend it
on ITL and see whether the gap closes.** The interval sweep in §6 is exactly that
experiment, but it should be scored as a *pair* — ITL against TTFT — and the pass
criterion is "ITL ≤ ATOM's while TTFT ≤ ATOM's", not "ITL improved".

If ITL reaches parity within the TTFT budget, the gap was a tuning choice and the
report should say so. If ITL plateaus above ATOM's while TTFT is already spent,
only then is there a genuine decode-execution deficit worth profiling — and that
is also when profiling a decode step becomes the right next move rather than a
first move.

It also predicts ATOM's flat TTFT has the same cause as its better ITL: deferring
prefill harder keeps the live working set smaller, which is precisely what our
c192 arm failed at. That makes one experiment test both threads.

## 7. Open questions

1. Is the c128 **+2.57 %** FP4 effect real? Needs a replicate; it is inside the
   noise floor.
1b. Does raising `--prefill-decode-interval` past 10 recover ITL, and at what
   throughput cost? This is the specific lever the dose-response knee points at.
2. Can bounding prefill work by *context* (not new tokens) recover ITL, and at
   what throughput cost?
3. What causes the OOR abort? Instrumentation now exists; cause open.
4. FP4 effect at c96/c160/c192 — no matched baselines exist.
5. c256, ATOM's best point, has never been run on SGLang here; the c192
   collapse suggests it is past the knee, so a matched comparison at ATOM's
   operating point may not be attainable without fixing the stalling first.

## 8. Tooling and operational notes

| tool | purpose |
|---|---|
| `arm_report.py <arm> [<baseline>]` | gates first, then headline; prints a noise verdict |
| `summary_table.py` | the table above; degrades to `[NO DATA]` on partial arms |
| `cache_tier_gate.py <arm>` | tier share, GPU-hit floor, ceiling shortfall |
| `decode_stall_split.py <arm>` | the §4 decode-vs-stall split |
| `wait_and_launch.sh <conc>` | shared-node safe launch + cleanup + gate |

- **The launcher never kills its own server.** Kill three shapes:
  `sglang.launch_server`, `sglang::tokenizer_worker:*`, `sglang::router`. A clean
  `rocm-smi` is not evidence the node is free — workers hold ports without VRAM.
- **Never `pkill -f` a pattern matching your own command line.** `pkill -f aiperf`
  killed a cleanup shell here.
- **This node is shared.** The c96 arm was lost once to an external manual
  cleanup at 05:12:25 while it was mid-run. Arm scripts from c96 onward refuse
  to start if any sglang/aiperf process exists, rather than killing blindly.
- **Never touch `/sgl-workspace/{sglang,aiter,mori}` while an arm runs.** A branch
  switch destroyed one arm; the FP4 adapter's aiter imports are lazy, so the tree
  is read for the whole run. Arms snapshot md5s into
  `TREE_CHECKSUMS_AT_START.txt` and verify them at the end.
- `OSL mean=1` in a workload-distribution block is **not** a broken run: aiperf's
  agentic warmup deliberately uses `max_tokens=1` to build cache pressure.
