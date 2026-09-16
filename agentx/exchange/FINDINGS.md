# FINDINGS — joint source of truth (B200 + MI355X)

Append only. Sign every block with node + date so a superseded number stays
identifiable. Detail and provenance live in the per-node `<node>-<topic>.md`
files; this file carries conclusions.

---

## ⚠ WITHDRAWN — the block below mixed DSPARK draft and full-model verify steps
*(mi355x, 2026-09-16, superseding its own entry from earlier the same day)*

Speculative decoding emits **two** `step[TARGET_VERIFY]` annotations per
`scheduler.run_batch`, at the **same `bs`**, so grouping by (type, bs) averaged a
4.0 ms draft step with a 74.0 ms full-model step. Corrected numbers, MI355X c128
pdi=24 steady state:

| | withdrawn | correct |
|---|---:|---|
| verify step wall p50 | 39.3 ms | **74.0 ms** full, 4.0 ms draft |
| `compute` p50 at bs=10 | 14.85 ms | **28.36 ms** |
| MoE calls/step | 32 | **61** full, 3 draft |

**"`compute` matches B200 within 1.4 %" is withdrawn.** MI355X's full-step
`compute` is 28.36 ms at bs=10 (24.94-38.91 across ranks bs 9-20).

**B200's published numbers are very likely affected the same way** — same tool,
same grouping, and B200's `n=38` verify steps at a single bs fit the same
2-per-batch pattern. Re-derive before either side quotes a ratio.

Resolved on the way: MoE calls/step **match** at 61 (one per layer); the earlier
"32 vs 61-64" discrepancy was the mixing artefact, so no renormalisation is
needed. Also, the "bimodal, quote p50 not mean" advice in both docs was
misdiagnosing this bug — split by class and each cluster is tight.

Fixed in `analysis/trace_common.py:verify_classes()` (splits by MoE call count,
not annotation order) and wired into `trace_summary.py`. `trace_ranks.py` is
**not** yet fixed and still mixes the classes.

What survives: the steady-state-vs-mid-ramp finding below, the stream-count
measurement (2-3 vs 132), the group-wide prefill barrier bound, and the
scheduler-log numbers, none of which depend on the split.

<details>
<summary>Withdrawn text, kept for traceability</summary>

## MI355X steady-state decode step: 39.3 ms — and `compute` nearly matches B200
*(mi355x, 2026-09-16 — detail in `mi355x-decode-trace.md`)*

At matched pdi=24, accept len (3.78 vs 3.770), per-request KV working set
(~151k vs ~152k) and 100 % cuda-graph replay, MI355X's steady-state
`TARGET_VERIFY` step wall is **39.3 ms p50 at bs=10**, captured at 167,538
tokens/request with pool usage plateaued. Flat across bs 9-20 (38.7-41.7 ms).

**No ratio is quoted against B200's 15.9 ms** — that number is retracted as
mid-ramp and B200 is re-capturing. 39.3 ms is simply the MI355X side.

**The most important number is not the wall, it is `compute`:** MI355X 14.85 ms
vs B200 15.06 ms at bs=10 — **within 1.4 %**. `compute` is barrier-free by
construction, so on present evidence the two platforms do comparable decode
compute per step, and the wall-clock gap lives in `barrier` (MI355X 21.8 ms,
~55 % of its step). That is the opposite of a kernel-efficiency story and should
be the next thing both nodes confirm, since the two captures are still not
like-for-like.

**Stream count differs sharply but is probably not the cause.** MI355X runs 2-3
streams against B200's 132, and its kernel sum equals its wall (1.00x overlap vs
B200's 1.54x). But B200's own 132->4 A/B cost only 4-5 % of step time, and
MI355X sits at essentially B200's single-stream setting — so stream count cannot
carry a gap of this size. Recorded as a measured difference, not a cause.

Open, and blocking a clean answer:

1. **Both captures are contaminated by the group-wide prefill barrier.** In the
   MI355X steady capture, ranks 0/1/3/6/7 have `TARGET_VERIFY` while **all eight**
   have `EXTEND`. Rank 6 is the extreme: barrier 174.69 ms of a 195.97 ms step.
   The two nodes need to report the same thing — either both filtered to steps
   with no concurrent `EXTEND` in the group, or both quoting `compute`.
2. **`gemm` is not comparable as bucketed.** B200 19.39 vs MI355X 9.34 ms/step
   p50 is probably naming: `deep_gemm::..._mega_moe_impl` matches `gemm` while
   `megamoe_stage1/2_compact` matches `moe` first. Use `gemm + moe` as one
   bucket until settled — B200 31.61 vs MI355X 43.21.

</details>

---

## The fixed 120 s settle is unsafe on both nodes — use a steady-state predicate
*(mi355x, 2026-09-16, confirming B200's retraction from its own data)*

MI355X's first capture used the prompted rule (first `done=` + 120 s) and landed
at 67,759 tokens/request, less than half steady state. Measured cost on one run:

| | mid-ramp | steady | delta |
|---|---:|---:|---:|
| per-request KV | 67,759 | 167,538 | 2.47x |
| step wall p50 | 28.3 ms | 39.3 ms | **+39 %** |
| `attn` p50 | 7.95 ms | 15.62 ms | +96 % |
| `moe` p50 | 24.23 ms | 33.87 ms | +40 % |

`attn` scaling with context is expected. **`moe` growing 40 % is not** — MoE work
is context-independent, so that is barrier absorbed into the fused all-to-all,
i.e. more waiting at steady state.

Replace the constant with a predicate on `server.log`: wait for per-request
`#full token` / `#running-req` to plateau (>= ~130k on these arms). A fixed sleep
is racing a ramp whose length depends on trace content and page-cache warmth.

---

## prefill_decode_interval was a launcher difference, and it is now matched
*(mi355x, 2026-09-16)*

`dsv4_fp4_mi355x_sglang_mtp.sh` now emits B200's values: pdi 24 (20 plus
`--balance-abs-threshold 32` at `CONC>=160`), `--load-balance-method
total_requests`, router `--policy cache_aware`. Every cross-platform ITL number
taken on MI355X before 2026-09-15 used pdi=10 and is not comparable.

Measured on MI355X at c128, the knob alone is a clean ITL<->TTFT trade, same
direction as B200's A/B: ITL p90 50.6 -> 33.9 ms (-33 %), TTFT p50 2.52 -> 4.01 s
(+59 %), and the same trade reproduces at c256 (ITL -30 %, TTFT p50 +187 %).

**The throughput half of that is not established.** tok/s/chip rose 6.6 % at
c128 and 4.7 % at c256, both inside the ~5 % replicate spread, and both are
partly explained by ISL mean drifting up (+2.6 % / +1.6 %) — the metric is ~99 %
input tokens, so longer prompts inflate it without the engine doing better.
Normalising to requests/s leaves +3.9 % / +3.0 %. What *is* solid is output
throughput, +8.3 % / +9.1 %, consistent with the ITL win.

---

## KV pool usage must not be matched across platforms
*(mi355x, 2026-09-16 — corrects a misreadable line in `analysis/METHOD.md`)*

`full token usage` is a fraction of each node's own pool capacity, and the pools
differ 5.4x (B200 2,217,472 tokens, MI355X ~12,074,667). Matching the fractions
would force genuinely different working sets. Match the **absolute** figure,
`#full token` / batch: B200 151,908 vs MI355X ~151,000 per request, i.e. aligned
within 3 %, while the fractions read 0.62 vs 0.15. METHOD.md's
"align kv pool usage" applies to partial-vs-complete runs on one node.

---

## Intermittent EPLB rebalance deadlock on MI355X
*(mi355x, 2026-09-16)*

Roughly 1 run in 3 on an EPLB arm hangs with a signature that defeats every
liveness check: `returned=` frozen, `errors=0`, `/metrics` answering 200,
schedulers alive, VRAM still allocated. Last server activity is
`Resetting ExpertDistributionRecorder...` from all 8 ranks; all 8 then hang in a
collective and the NCCL watchdog kills the process 600 s later with
`c10::DistBackendError`. Nothing reaches launcher stdout. Judge liveness only by
`returned=`/`done=` moving. A straight retry cleared it.

---

## RESOLVED: the gap is inside the decode step, and it is not compute
*(b200, 2026-09-16, against mi355x's steady-state capture the same day)*

Both nodes now have a steady-state, class-split capture at pdi=24 with matched
KV working sets (**B200 165,220 tok/req vs MI355X 167,538, 1.4 % apart**). This
is the first genuinely controlled comparison; every earlier one mismatched the
ramp position, the step class, or both.

| per full-model verify step | B200 (bs=12) | MI355X (bs=10) | ratio |
|---|---:|---:|---:|
| **step wall p50** | **30.0 ms** | **74.0 ms** | **2.47x** |
| draft step wall | 2.0 | 4.0 | 2.0x |
| summed kernel | 50.18 | 75.24 | 1.50x |
| `compute` (attn+gemm+quant+norm_rope+sample) | 31.47 | 28.37 | **0.90x** |
| `barrier` (moe+comm) | 14.77 | 40.85 | 2.77x |
| summed kernel / wall (overlap) | **1.67x** | **1.00x** | |

### 1. The discriminator is answered — and it inverts the earlier reading

| | B200 | MI355X | ratio |
|---|---:|---:|---:|
| log-implied step (scheduler) | 79.2 ms | 121.76 ms | 1.54x |
| decode step (full + draft) | 32 ms | 78 ms | **2.44x** |
| remainder (prefill, gaps) | 47.2 ms | 43.8 ms | **0.93x** |

The decode step accounts for **all** of the 42.6 ms log-implied gap — 46 ms of
it — while the portion outside the decode step is slightly *better* on MI355X.
**The prefill barrier is not where the gap lives.** The old "only 22 % of B200's
wall is inside decode steps" pointed the other way and was an artefact of the
retracted 15.9 ms.

### 2. But it is not "decode kernels are slower" either

`compute` is **28.37 ms on MI355X against 31.47 ms on B200** — MI355X is 10 %
*faster* on the barrier-free kernels, at a smaller bs. The entire decode-step
gap sits in `barrier`, which on both nodes absorbs the DP group wait inside the
fused MoE all-to-all (`compute + barrier` is near-constant per rank while
`compute` alone varies: ~46-49 ms on B200, ~69-75 ms on MI355X).

So the question has moved from "kernels or prefill" to **"why does MI355X's
group wait cost 2.8x B200's"**, with two measured candidates below.

### 3. Candidate A — DP load imbalance: NOT ESTABLISHED, and the earlier
### framing of it was invalid

An earlier version of this block compared a B200 `compute` spread of 0.6 ms
against an MI355X spread of 14 ms (24.9-38.9) and called the difference
imbalance. **That comparison does not hold: each rank sits at its own `bs`, so
rank and batch size move together and neither table separates them.**

What *is* established, because B200's capture happens to repeat several `bs`
values across different ranks — **at fixed `bs`, B200's cross-rank spread is
0.0-0.3 ms**:

| bs | ranks | `compute` p50 | spread |
|---:|---|---|---:|
| 9 | TP-4 / TP-6 / TP-7 | 31.48 / 31.16 / 31.36 | 0.32 |
| 10 | TP-4 / TP-7 | 31.45 / 31.60 | 0.15 |
| 11 | TP-0 / TP-4 | 31.38 / 31.29 | 0.09 |
| 12 | TP-0 / TP-4 | 31.47 / 31.48 | 0.01 |

MI355X's published table has one `bs` per rank and no repeats, so the matching
statement cannot be made there. **Until it is, there is no measured claim that
MI355X's imbalance exceeds B200's.**

To settle it, MI355X should report `compute` for two or more ranks *at the same
`bs`* within one capture, or bucket `compute` by bs the way `decode_stats.py`
buckets `step_ms`.

### 3b. What the overlapping batch sizes do establish

`bs` 9 and 10 appear on both nodes, which makes these two rows controlled:

| bs | B200 `compute` | MI355X `compute` | MI355X vs B200 |
|---:|---|---:|---:|
| 9 | 31.16-31.48 | 24.94 | **0.79-0.80x** |
| 10 | 31.45-31.60 | 28.36 | **0.90x** |

So "MI355X's barrier-free compute is lower than B200's" **survives matching on
bs, and is slightly stronger than the 0.90x quoted from the unmatched bs=12 vs
bs=10 pair.** The headline conclusion — the gap is `barrier`, not `compute` —
does not depend on the imbalance claim withdrawn above.

**Hypothesis, two points only, do not act on it yet:** B200's `compute` is flat
in batch (31.03 at bs=1 to 31.47 at bs=12, i.e. fixed-cost dominated) while
MI355X's rises 24.94 -> 28.36 from bs 9 to 10 (+13.7 %). If that slope held, the
curves would cross near bs 11-12 — and MI355X's `running-req/rank` p50 is 12.
But MI355X's own table is non-monotonic beyond bs=14 (16 -> 35.25,
17-20 -> 28.87, the latter a p50 over mixed bs), so the slope is not real yet.
The test is a `compute`-vs-bs curve per node, not two points.

### 4. Candidate B — the "stream overlap is worth 4-5 %" bound does NOT apply here

**This retracts how both docs used that A/B.** B200 fits 50.18 ms of kernels
into a 30.0 ms wall (**1.67x**); MI355X fits 75.24 into 74.0 (**1.00x**). The
B200 flag A/B that measured 4-5 % moved the ratio only **1.49x -> 1.33x** (132
streams -> 4) — it never reached 1.00x, so it bounds the cost of *fewer*
streams, not of *no overlap at all*. MI355X sits outside the range that
experiment explored, and citing 4-5 % to dismiss overlap is unsupported.

The wall gap decomposes cleanly into the two candidates:
**1.50x (more kernel time) x 1.67x (less overlap) = 2.50x, against 2.47x
measured.**

### 5. What each side should do next

- **MI355X:** publish `compute` for two or more ranks **at the same `bs`**, so
  rank and batch stop moving together — that is what §3 needs and it needs no
  new run, only a regroup of the existing capture. Then a `compute`-vs-bs curve
  for §3b. And establish whether any kernel overlap is reachable at all on ROCm,
  since 1.00x is the single largest multiplier in the table.
- **B200:** batch ranges barely overlap (B200 bs 1-12, MI355X 9-20, common
  ground only at 9 and 10). Re-capture at a larger batch to extend the
  `compute`-vs-bs curve into MI355X's operating range.
- **Both:** always publish step wall, its `bs`, *and* the capture window's KV
  working set together. Two of the three were missing from every capture before
  today, and that is what made three separate numbers wrong.
