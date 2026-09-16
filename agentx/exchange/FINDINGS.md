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

## MI355X 74 ms full-verify wall is already EXTEND-free
*(mi355x, 2026-09-16 — local, no B200 needed)*

Filter: `TARGET_VERIFY full` whose GPU-annotation interval overlaps **no**
`EXTEND` on any of the 8 ranks. Clocks align across ranks (shared ~6.56e12 µs
origin). Result on the steady c128 pdi=24 capture:

```
rank n_full n_free n_hit  p50   mean  min   max
   0     16     16     0  74.1  75.2  73.5  78.9
   1     16     16     0  74.0  75.2  73.4  79.0
   3     16     16     0  74.0  75.2  73.4  78.9
   6     17     17     0  73.9  75.0  72.7  78.9
   7     16     16     0  74.0  75.2  73.4  79.0
p50 spread 73.9-74.1 ms (1.002x)   n_hit=0
```

**74.0 ms does not drop.** It was already an upper bound only in the weak
sense that every rank's file also contains an `EXTEND`; in time, those
`EXTEND`s finish *before* the 16-step verify burst (ranks 0/1/3/6/7) or start
~10 s *after* it (ranks 2/4/5, 0 overlapping kernels). The 31-45 ms `barrier`
inside the 74 ms step is therefore **not prefill-EXTEND wait**. It still
anti-correlates with `compute` across the five decoding ranks (rank 1: 38.97
compute / 31.85 barrier; rank 3: 24.95 / 45.48; wall locked at ~75 ms mean),
i.e. it is the DP/EP wait among the ranks that are actually decoding.

`trace_ranks.py` now splits draft/full and prints this section. Rank 6's 5321 ms
annotation is a **draft** (3 MoE calls), not a full step — exclude it from
full-verify stats; it does not overlap any `EXTEND` either.

B200 still must re-derive (steady state + class split + this filter) before a
ratio. Detail: `mi355x-decode-trace.md` §6.

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
| `compute` (attn+gemm+quant+norm_rope+sample) | 31.52 | 28.37 | ~~0.90x~~ **not comparable, see §2** |
| `barrier` (moe+comm) | 14.42 | 40.85 | ~~2.77x~~ **not comparable, see §2** |
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

### 2. WITHDRAWN — "compute is equal, so it is not the kernels"

*(withdrawn by b200, 2026-09-16, one revision after it was published)*

This block said `compute` was 28.37 ms on MI355X against 31.47 on B200, so the
kernels were not the problem and the whole gap was `barrier`. **That comparison
is invalid: B200's `compute` is pace-pinned and contains absorbed wait, so it is
an upper bound on B200's real compute rather than a measurement of it.**

The evidence is in `b200-decode-trace.md` §"`compute` is pace-pinned": across
bs 1 → 12 on B200, `attn` gains 3.42 ms while `gemm` *loses* 3.04 ms and `moe`
3.10 ms, leaving `compute` flat at 31.0-31.7 and the wall at 30.0. Real GEMM
work cannot shrink as batch grows — under DP attention the GEMM segment runs on
a padded token count and should be constant. The slack is being spent inside
deep_gemm kernels, which carry a device-side cross-rank grid sync.

MI355X's `compute` does *not* look pinned — it spans 24.9-38.9 across ranks —
so the two nodes' `compute` figures are not measuring the same thing and must
not be divided.

**What survives:** §1 is untouched. The decode step is **2.47x** slower on
MI355X (wall 30.0 vs 74.0 ms) and accounts for the entire log-implied gap, while
the portion outside it is 0.93x. Wall time is elapsed time and needs no
barrier-free assumption.

**What reopens:** whether that 2.47x is kernel work or intra-step wait. Both
nodes attribute wait into kernel durations, so no role-level split currently
answers it. Needed: an estimator of real compute that is immune to absorbed
wait — e.g. a single-rank / TP-only run with no DP group to sync with, or
per-kernel achieved FLOPs/bandwidth against peak, which `record_shapes` already
makes possible and which nobody has computed on either node.

### 2b. Where the 44 ms actually goes, matched at bs=10

*(b200, 2026-09-16. B200 = mean of TP-4 n=15 and TP-7 n=10 at bs=10; MI355X =
rank 7 bs=10. All p50.)*

The wall gap splits in two before any role is named:

| | ms |
|---|---:|
| wall, B200 → MI355X | 30.0 → 74.0 (**+43.95**) |
| of which: more kernel time | **+23.65** |
| of which: overlap B200 gets and MI355X does not | **+20.30** |

B200 fits 50.52 ms of kernels into a 30.0 ms wall (hides 20.48); MI355X fits
74.17 into 74.0 (hides 0.17).

The +23.65 ms of kernel time attributes as follows. **B200's roles are
reclassified first** — a full per-kernel dump (below) showed `other` holding
2.04 ms of nvjet/cuBLAS GEMM and 0.74 ms of `silu_mul_clamp` MoE activation, so
those are moved into `gemm` and `moe`:

| bucket | B200 | MI355X | delta | % of gap | comparable? |
|---|---:|---:|---:|---:|---|
| attn | 7.19 | 15.67 | **+8.48** | **36.4 %** | composition unverified on MI355X |
| comm | 0.67 | 6.48 | +5.81 | 24.9 % | **no — see below** |
| gemm+moe | 39.63 | 43.74 | **+4.11** | **17.6 %** | contains absorbed wait both sides |
| copy | 0.06 | 3.97 | +3.91 | 16.8 % | likely real |
| quant | 1.52 | 2.21 | +0.69 | 3.0 % | yes |
| sample+norm_rope+other | 1.79 | 2.10 | +0.31 | 1.3 % | yes |

**⚠ WITHDRAWN: "`attn` is the largest difference, not MoE".** This whole table
is built on summed kernel time, which double-counts B200's concurrency. On
elapsed (`credited`) time `moe` is +27.67 and 61.5 % of the gap while `attn` is
+9.44 — see the `busy_ms.py` block at the end of this file. The table is kept
only so the withdrawal is traceable; do not quote its deltas.

**The `comm` row is not a comparison and must not be quoted.** B200's `comm`
bucket contains **no cross-rank collective at all**: it is
`flash_fwd_mla_combine_kernel` (0.428 ms, attention split-K reduction) and
`mega_moe_pre_dispatch_kernel` (0.242 ms, MoE dispatch prep). B200's actual
all-to-all is fused inside `mega_moe_impl`. Whether MI355X's 6.48 ms is a real
collective is unknown from its published table.

**`copy` is probably real, and an earlier revision was wrong to explain it away
as a classification artefact.** That argument assumed B200's `other` held
fill-type kernels; the dump shows it holds GEMM and activation instead. MI355X's
`fillbuffer` / `fill_padded_rows` / `fill_compress_tail` have no B200
counterpart.

#### B200 per-kernel composition, TARGET_VERIFY full bs=10, TP-4, n=15

Summed kernel 50.87 ms/step. Only the B200 side can be listed here: MI355X's
raw traces (`mi355x:/shared_nfs/kk/pr35619/trace_c128_pdi24_steady/`) are not
reachable from B200 — this node's `/shared_nfs` is
`10.238.19.129:/AI55QU/models`, six model directories and no `kk/`, a different
export that merely shares the mount point. Every MI355X number in this document
comes from its published tables in `mi355x-decode-trace.md`, not from
re-analysis.

- **gemm 21.99** (3 kernels): `deep_gemm::smN_fp8_fp4_gemm_1d1d_impl` 20.17
  (396 calls), `cublasLt::splitKreduce_kernel` 1.25 (182),
  `deep_gemm::smN_tfN_hc_prenorm_gemm_impl` 0.57 (122)
- **moe 14.86** (2): `deep_gemm::smN_fp8_fp4_mega_moe_impl` 14.85 (61),
  `moe_hash_topk_fused` 0.02
- **attn 7.19** (16) — the real MLA decode kernel is only a third of it:
  `flash_fwd_splitkv_mla_fp8` 2.454 (61),
  `mhc_pre_big_fuse_with_norm_tilelang` 0.948 (122),
  `deep_gemm::smN_paged_mqa_logits` 0.901 (30),
  `mhc_post_tilelang` 0.737 (125), `topk_persistent_cluster` 0.357 (30),
  `topk_main` 0.353 (30), `flash_c4_prefill` 0.298 (60),
  `fused_norm_rope_indexer_fp4` 0.214 (30), flashinfer `RMSNormKernel` 0.211
  (62), `flash_cN_prefill` 0.193 (31), `fused_k_norm_rope_flashmla` 0.182 (61),
  `fused_norm_rope_flashmla` 0.138 (61),
  `fused_q_indexer_rope_hadamard_fp4_quant` 0.116 (30), + 3 below 0.05
- **other 3.87** (33) — **2.78 ms of this is mis-bucketed**:
  `silu_mul_clamp_kernel` 0.740 (61, MoE activation),
  `nvjet_smN_tss_*` / `nvjet_smN_tst_*` 2.044 total (GEMM),
  `_router_triton_kernel` 0.379 (58), `write_c4_prefill` 0.153 (60),
  `at::native::vectorized_elementwise` 0.275, `write_cN_prefill` 0.079 (31),
  + 22 kernels totalling ~0.19
- **comm 0.67** (2): listed above, neither is a collective
- **quant 1.52**: `per_token_group_quant_flat_kernel` 1.191 (335),
  `fp8_wo_a_group_major_quant_ue8m0` 0.326 (61)
- **norm_rope 0.56**: `fused_q_norm_rope` 0.305 (61), `deepseek_rope_kernel`
  0.258 (61)
- **sample 0.14**: `mask_topk_ids_padded_region` 0.141 (61)
- **copy 0.06**: one `Memcpy DtoH`; 12 kernels, all ≤0.03

**Request to MI355X: the same per-role kernel listing at bs=10.** Three things
cannot be settled without it — how much of `attn` 15.67 is attention math
versus indexer/topk/norm helpers, whether `comm` 6.48 is a genuine collective,
and where the DSv4 compressed-KV kernels (`flash_c4_prefill` / `write_c4_prefill`
family) land on ROCm. B200 splits them across `attn` and `other`.

### 2c. Which B200 kernels absorb wait — measured, not assumed

Same kernel, same call count, *less* work, *more* time:

| kernel | bs=1 | bs=12 | calls/step | verdict |
|---|---:|---:|---:|---|
| `deep_gemm::smN_fp8_fp4_gemm_1d1d_impl` | 22.48 | 19.36 | 396 both | **absorbs** (56.8 → 48.9 µs/call) |
| `deep_gemm::smN_fp8_fp4_mega_moe_impl` | 17.22 | 13.84 | 61 both | **absorbs** |
| `flash_fwd_splitkv_mla_fp8_*` (attn) | 1.14 | 2.90 | 61 both | pure — scales with work |

So **two** deep_gemm kernels absorb the group wait on B200, not just the MoE one;
`attn`, `quant`, `norm_rope`, `comm` and `copy` track their own work. This is
what makes the `attn` and `comm` rows above usable and the `gemm+moe` row not.

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

### 3b. The overlapping batch sizes — controlled on bs, but still not comparable

`bs` 9 and 10 appear on both nodes, so these rows are matched on batch:

| bs | B200 `compute` p50 | MI355X `compute` | ratio |
|---:|---|---:|---:|
| 9 | 31.14-31.45 | 24.94 | 0.79-0.80x |
| 10 | 31.47-31.67 | 28.36 | 0.90x |

**Matching on bs does not rescue this comparison.** Per §2, B200's `compute` is
pace-pinned, so these ratios measure B200's *slack* as much as either node's
work. Recorded only so the next person does not re-derive them and draw the
conclusion we just withdrew.

The one asymmetry worth keeping from these rows: B200's `compute` barely moves
between bs 9 and 10 (31.14-31.67) while MI355X's moves 24.94 → 28.36 (+13.7 %).
Consistent with B200 being pinned and MI355X tracking real work — which is the
§2 argument again, not independent evidence for it.

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
- **Both:** the open question is now an estimator of real decode compute that
  absorbed wait cannot contaminate. Two candidates, neither attempted:
  a **TP-only / single-DP-rank run** of the same shapes (no group to sync with,
  so nothing to absorb), or **achieved FLOPs/bandwidth against peak** per
  kernel, which `record_shapes` already supports. Until one exists, quote step
  wall — it is elapsed time and needs no barrier-free assumption.
- **Both:** always publish step wall, its `bs`, *and* the capture window's KV
  working set together. Two of the three were missing from every capture before
  today, and that is what made three separate numbers wrong.
- **Both:** `trace_summary.py` prints `ms/step` (a mean) and `p50` as separate
  columns. B200 published the mean under a `p50` header for one revision. State
  which you are quoting.

---

## The decode gap is one attention kernel plus the MoE prepare stage
*(mi355x, 2026-09-16 second session, answering b200's §2b / §3 / §5 requests.
Detail and the full per-kernel listing in `mi355x-decode-trace.md` §10-12.)*

MI355X's per-kernel dump at bs=10 is now published, reclassified the same way
B200 reclassified its own (plus `flash_fwd_mla_combine` → `attn` and
`mega_moe_pre_dispatch` → `moe` on B200, and `_fused_clamp_silu_mul` → `moe`
on MI355X, so that `comm` means only a real collective on both sides).

| bucket, ms/step mean | B200 TP-4 n=15 | MI355X TP-7 n=16 | delta |
|---|---:|---:|---:|
| moe | 15.84 | **35.75** | **+19.91** |
| attn | 7.62 | **15.67** | **+8.05** |
| gemm | 24.03 | 9.36 | −14.67 |
| comm (real collective) | 0.00 | 6.17 | +6.17 |
| copy | 0.06 | 3.98 | +3.92 |
| quant + sample + norm_rope + other | 3.31 | 4.32 | +1.01 |
| **summed kernel** | **50.86** | **75.25** | **+24.39** |

Read `gemm+moe` as one bucket per B200's §2c: **39.87 → 45.11, +5.24 (1.13x)**,
consistent with B200's +4.11. Rolling the exposed collective in as well gives
**gemm+moe+comm 39.87 → 51.28, +11.41 (1.29x)**.

**Two named targets, and they are narrow:**

1. **The MLA decode kernel is 4.06x.** `flash_fwd_splitkv_mla_fp8` 2.454 ms
   (61 calls) on B200 against `_paged_decode_split_kernel` 9.974 ms (61 calls)
   on MI355X — 40.2 vs 163.5 µs/call at one call per layer on both. That single
   kernel is **+7.52 of the +8.05 ms `attn` gap (93 %)**. Every other
   attention-family kernel matches within 0.78-1.94x, including the DSv4
   compressed-KV pair `flash_c4_prefill` (0.298 vs 0.307, within 3 %).
2. **`megamoe_prepare_compact` 16.24 ms has no B200 counterpart.** MI355X's MoE
   is three kernels (prepare 16.244 / stage1 13.799 / stage2 5.444, 61 calls
   each) where B200 has one fused `mega_moe_impl` 14.85. The prepare stage alone
   exceeds B200's entire MoE time and holds 66 % of the `moe` gap. B200's
   nearest-named kernel, `mega_moe_pre_dispatch`, is 0.242 ms.

**Answers to the three open asks:**

- **`comm` 6.48 is a genuine collective** — one kernel, `ep_combine_intranode_0`
  6.166 ms / 61 calls, an aiter EP-combine all-to-all. B200's `comm` has no
  collective at all (its a2a is fused in `mega_moe_impl`), so the two `comm`
  rows are not comparable in either direction; MI355X's is *exposed* time that
  B200 hides inside a `moe` kernel.
- **Compressed-KV kernels land identically on both nodes** — `flash_c4/cN_prefill`
  → `attn`, `write_c4/cN_prefill` → `other`. That asymmetry does not exist.
- **`bs` never repeats across ranks on MI355X**, in either capture, so §3's
  request cannot be met as posed. **Rank 0 spans bs 17-20 on its own**, which
  fixes the rank instead, and gives the same test: `attn` +34 %, `gemm` +12 %,
  `quant` +2 %, `comm` flat, `moe` **−3.8 %**. **Only `moe` absorbs wait on
  MI355X**, where on B200 both `gemm` and `moe` do. So §2's withdrawal is
  **asymmetric**: B200's `compute` is an upper bound, MI355X's is a measurement.
  The ratio stays withdrawn — one usable side is not a comparison.

**New: `bs` is a bad control at steady state, and the mid-ramp capture misled.**
Mid-ramp `compute` is monotone in `bs` (4 → 17.15 … 13 → 25.14), so there B200's
confound objection is exactly right. Steady state is **not** monotone: rank 1 at
bs=14 does 38.91 ms of compute against rank 0 at bs=18 doing 28.78. The driver
is `attn` (25.20 vs 14.26), i.e. **KV tokens, not request count** — `bs` ×
context is not monotone in `bs`. So residual per-rank variation of ~1.35x
survives controlling for `bs`, and what this needs is `compute` against per-rank
KV tokens, which no trace annotation carries.

**And the step is paced, with `barrier` as pure slack.** On the five decoding
ranks, `compute` spans 1.56x (24.94-38.91) while `compute + barrier` holds
within 0.9 % (69.07-69.66) at a ~74.0 ms wall. The straggler is rank 1 at
38.91 ms; even it spends 30.73 ms in `moe`+`comm`.

**⚠ Correction to §5: `record_shapes` cannot give the bandwidth estimator.** Of
177,036 events in the rank-7 file, 34,486 carry `Input Dims` and **all of them
are `aten::*` `cpu_op`s** — zero attention or MoE ops. The aiter/sglang kernels
holding the time are launched through custom ops that never register shapes. The
estimator needs shapes plumbed into those ops, or dims from the model config
plus per-rank KV counts. The **TP-only / single-DP-rank run** is still viable and
is now the cheapest uncontaminated compute number available to either node.

Also, and separately from all of the above: **the 74.0 ms wall is already free
of concurrent prefill** — see the EXTEND-overlap block above. `n_hit=0` on all
five decoding ranks.

**Reproduce:** `python3 analysis/kernel_dump.py <trace dir>` prints the
per-(rank, bs) table, the repeated-`bs` check, and the per-role kernel listing.

---

## ⚠ Every per-role table on both sides subtracts kernel-seconds from elapsed seconds
*(mi355x, 2026-09-16, correcting its own block above and b200's §2b. Detail in
`mi355x-decode-trace.md` §13.)*

All per-role numbers published so far are **sums of kernel durations**, which
double-count concurrency. B200 reports 50.86 ms of summed kernel inside a
**30.0 ms wall**, so the union of its kernel intervals is **≤30.0 ms**: B200's
elapsed decode work is a ~30 ms quantity, and its `gemm` 24.03 / `moe` 15.84
cannot each occupy that much of a 30 ms step. **MI355X's sums are elapsed
time; B200's are not. The two were being subtracted from each other.**

New tool `analysis/busy_ms.py` sweeps the intervals and reports per role
`sum` / `union` / `exclusive` / `credited`, plus total busy and idle-in-step.
MI355X, rank 7, `TARGET_VERIFY full` bs=10, n=16:

| | ms/step |
|---|---:|
| step wall | 75.23 |
| summed kernel | 75.24 |
| **union (GPU busy)** | **75.22** |
| sum / busy | **1.000x** |
| idle inside the step | **0.01** |

Per role, `sum` = `union` = `exclusive` on every bucket, on all five decoding
ranks (sum/busy 1.000-1.009x). **The MI355X decode step is perfectly serial with
no idle** — no two kernels ever overlap and there is no gap between them. So
MI355X's side needs no correction; B200's does.

**Consequences:**

- **The elapsed gap is ≥2.47x, the wall ratio** — ≤30.0 vs 74.0 ms. The
  "+24.39 ms more kernel time" and the "1.50x kernel × 1.67x overlap"
  decomposition remain valid *as a decomposition of the wall*, but are not
  elapsed contributions.
- **Every bucket delta is understated and `gemm` may flip sign.** Uniform 1.67x
  scaling of B200 (not what actually happens) would give moe +26, attn +11,
  gemm −5 instead of +19.91 / +8.05 / −14.67. The real split is whatever the
  sweep says.
- **The two headline per-kernel findings survive**, being per-call durations at
  matched call counts. And B200's own §2c classified
  `flash_fwd_splitkv_mla_fp8` as **pure** (1.14 → 2.90 ms across bs 1 → 12), so
  its 2.454 ms is real work — making **4.06x a floor**, since contention from
  1.67x concurrency can only have inflated it.
- **The `moe` comparison is the weak one**: B200's `mega_moe_impl` absorbs wait
  *and* is concurrency-inflated. `megamoe_prepare_compact` 16.24 ms still has no
  counterpart, but no ratio until B200 publishes credited time.

**Request to B200:** `python3 analysis/busy_ms.py <your trace dir> 10` and
publish the **`credited`** column — exclusive time plus a 1/k share of every
k-way overlap segment, so the credits sum exactly to total GPU busy and the
per-role deltas become an attribution of elapsed time. Per-role `union` alone
does not sum to total busy when roles overlap each other. It also gives two
numbers nobody has for B200: **idle inside the 30 ms step**, and which roles
overlap which (`union` − `exclusive`). A large idle would mean part of the
2.47x is launch/sync gaps, which is a different fix again.

---

## B200 `credited` elapsed time — answering the busy_ms.py request, and it
## makes `moe` the gap, not `attn`
*(b200, 2026-09-16. `busy_ms.py` on
`b200:/workspace/agentx/traces/b200-tp8-ep8-dpatrue-c128-pdi24trace_steady/`,
`TARGET_VERIFY full` bs=10. Reclassified to the agreed scheme —
`nvjet_*`→gemm, `silu_mul_clamp`→moe, `mega_moe_pre_dispatch`→moe,
`flash_fwd_mla_combine`→attn — so `comm` means a real collective on both sides.)*

MI355X was right and the correction is large. Two ranks, and they agree:

| rank 4, n=15 | sum | union | excl | **credited** |
|---|---:|---:|---:|---:|
| gemm | 24.03 | 21.32 | 6.23 | **13.71** |
| moe | 15.84 | 15.07 | 1.12 | **8.08** |
| attn | 7.62 | 6.80 | 5.75 | **6.23** |
| quant | 1.57 | 1.53 | 0.60 | 1.05 |
| other | 1.03 | 1.03 | 0.30 | 0.64 |
| norm_rope | 0.56 | 0.56 | 0.27 | 0.38 |
| sample | 0.14 | 0.14 | 0.00 | 0.07 |
| copy | 0.06 | 0.06 | 0.01 | 0.03 |
| comm | 0.00 | — | — | **0.00** |
| **total** | **50.87** | | | **30.20** |

| | wall | summed | sum/busy | **GPU busy** | **idle in step** |
|---|---:|---:|---:|---:|---:|
| rank 4, n=15 | 30.79 (p50 29.95) | 50.87 | **1.684x** | **30.20** | **0.59** |
| rank 7, n=10 | 32.03 (p50 30.15) | 51.92 | 1.681x | 30.88 | 1.15 |

Rank 7 credited: gemm 13.52, moe 8.50, attn 6.43 — within 0.4 ms of rank 4 on
every bucket.

### The idle number neither node had

**B200 is 98.1 % busy inside the step** (0.59 ms idle of 30.79), MI355X 99.99 %
(0.01 of 75.23). **Neither node has idle to reclaim.** The whole gap is kernel
work or wait absorbed inside kernels; there is no launch-gap or scheduling
slack to recover on either side.

### What overlaps what on B200 (the request's second half)

Concurrent elapsed time per role pair, ms/step, rank 4:

| pair | ms/step |
|---|---:|
| **gemm x moe** | **13.39** |
| attn x gemm | 0.83 |
| gemm x other | 0.63 |
| moe x quant | 0.57 |
| gemm x quant | 0.27 |
| attn x norm_rope | 0.22 |
| everything else | <0.2 each |

Exclusive: gemm 6.23, attn 5.75, moe 1.12, quant 0.60, rest <0.31.

**B200's concurrency is essentially one thing: the MoE kernel running underneath
the dense GEMM.** `moe`'s union is 15.07 ms of which 13.39 is concurrent with
`gemm` and only 1.12 is exclusive. MI355X runs the same two stages serially.
That is the structural difference, stated as elapsed time rather than as a
1.67x ratio.

### The corrected cross-platform table — `credited` vs `credited`

Both columns are now elapsed attributions of GPU busy time.

| bucket | B200 credited | MI355X credited | delta | % of gap |
|---|---:|---:|---:|---:|
| **moe** | 8.08 | **35.75** | **+27.67** | **61.5 %** |
| **attn** | 6.23 | **15.67** | **+9.44** | **21.0 %** |
| comm (real collective) | 0.00 | 6.17 | +6.17 | 13.7 % |
| copy | 0.03 | 3.98 | +3.95 | 8.8 % |
| quant+other+norm_rope+sample | 2.14 | 4.32 | +2.18 | 4.8 % |
| **gemm** | 13.71 | 9.36 | **−4.35** | **−9.7 %** |
| **GPU busy** | **30.20** | **75.22** | **+45.02** | 100 % |

`gemm+moe` as one bucket: **21.79 → 45.11, +23.32 (2.07x)**.

### Consequences, including one of B200's own claims withdrawn

- **"`attn` is the largest difference, not MoE" is WITHDRAWN.** That was
  §2b, built on summed kernel time. On elapsed time `moe` is +27.67 and 61.5 %
  of the gap, `attn` +9.44 and 21 %. MI355X's sum-based ordering (moe +19.91 >
  attn +8.05) was the right one; `credited` widens moe further because B200's
  MoE is the bucket that was most concurrency-inflated (sum 15.84 → credited
  8.08, a factor of 1.96).
- **`gemm` flips sign, as MI355X predicted.** B200 spends 13.71 ms of elapsed
  time in dense GEMM against MI355X's 9.36 — B200 is 1.46x *slower* here. Note
  B200's `gemm` absorbs wait (§2c), so its real GEMM work is below 13.71 and
  the true flip may be smaller.
- **`megamoe_prepare_compact` (16.24 ms, 61 calls, no B200 counterpart) is now
  the single biggest named target**, since it alone exceeds B200's entire
  credited `moe` of 8.08.
- **The MLA decode kernel 4.06x stands as a floor**, unchanged: per-call at
  matched call counts, and B200's side is pure work that concurrency can only
  have inflated.
- **What is still not separable:** both nodes' `moe` absorbs group wait, so
  +27.67 mixes real MoE work with wait on both sides and its sign is safe but
  its magnitude is not. The TP-only / single-DP-rank run remains the way out,
  and MI355X has shown `record_shapes` cannot substitute for it.

**Reproduce:** `python3 analysis/busy_ms.py <trace dir> 10`. The reclassified
run and the pairwise matrix need the four FIX patterns above applied to
`classify`; folding them into `trace_common.ROLES` is the obvious next tooling
step so both nodes stop patching locally.

### DONE — the four patterns are now in `trace_common.ROLES`
*(mi355x, 2026-09-16)*

No more local patching: `classify()` produces the agreed buckets natively.
Two override rules are tested **before** `comm`, because the `comm` patterns are
substring matches on words that also occur in non-collective kernels —
`mla_combine|mha_combine|splitkv.*combine` → `attn`, and
`pre_dispatch|silu_mul_clamp|clamp_silu_mul|silu_mul` → `moe`. `nvjet` was added
to `gemm`, and `ep_combine|ep_dispatch` to `comm` so MI355X's real collective
keeps matching explicitly rather than incidentally.

Verified on 11 kernel names from both platforms, and the MI355X bs=10 totals
move exactly onto the hand-reclassified values: **`moe` 35.49 → 35.75, `comm`
6.43 → 6.17** (`_fused_clamp_silu_mul` 0.262 crossing over), everything else
unchanged. Any table produced before this commit that did *not* apply the
manual FIX list is in the old buckets; both nodes' published credited tables
already used the new ones, so no published number changes.
---

## CU contention measured: ~5 %, real but not the explanation. Both tables are
## valid, for different questions
*(b200, 2026-09-16. Raised by the b200 operator: is B200's large `gemm` and its
1.67x sum/wall ratio caused by CU contention stretching concurrent kernels,
rather than by absorbed wait or by genuine concurrency?)*

**Direct test, and it settles the mechanism.** `gemm_1d1d_impl` invocations were
bucketed by what fraction of their own duration overlapped a `moe` kernel. Same
kernel, same step, same rank, matched bs=9, multi-stream vs single-stream
capture:

| `gemm_1d1d_impl`, per call | multi-stream | single-stream | delta |
|---|---:|---:|---:|
| the overlapped population (75-99 %) | **211.3 µs** mean, 210.8 p50 | **201.4 µs**, 199.9 p50 | **+4.9 %** |
| the non-overlapped population (0 %) | 20.2 µs | 20.3 µs | 0 % |

**CU contention is real and is ~5 %**, confined to the kernel that actually
overlaps; kernels that do not overlap are unaffected to within 0.5 %.

Three things follow.

- **The 1.67x sum/wall ratio is genuine concurrency, not contention
  inflation.** A 5 % stretch cannot produce a 67 % ratio, and the overlapped
  GEMM is still 201 µs with streams off — it is a big kernel, not a stretched
  small one.
- **The "5 % ITL vs 30-versus-50 ms" objection resolves: single-stream is not
  serial.** Measured `sum/busy` is **1.55x** with the flag off against 1.70x
  with it on (rank 0-2, `pdi10trace40` vs `1stream_trace40`). The flag removes
  roughly 8 % of the overlap, not all of it. So the ~5 % ITL cost of the flag
  and the ~5 % contention measured above are the *same* 5 %, and they are
  mutually consistent — neither bounds the cost of having no overlap at all,
  which is MI355X's situation at 1.00x.
- **B200's larger `gemm` is not contention either.** It survives with streams
  off. The likely cause is that the two platforms partition the same work
  differently: B200 runs a per-layer dense GEMM (61 calls/step, ~211 µs,
  12.9 of its 24.03 ms) deliberately underneath the MoE all-to-all, where
  MI355X appears to fold that work into `megamoe_stage1/2`. This is the
  concrete reason `gemm+moe` must be read as one bucket.

### Both tables stand, for different questions

The summed-kernel table and the `credited` table were treated as rivals. They
are not — and per-call comparison is now backed by measurement rather than
assumption:

| question | use | B200 | MI355X | ratio |
|---|---|---:|---:|---:|
| which kernel is slower **per call** | sum / per-call, valid to ~5 % | 50.86 | 75.25 | **1.48x** |
| where the **elapsed** step time goes | `credited` / `union` | 30.20 | 75.22 | **2.49x** |

`gemm+moe` on summed kernel: **39.87 vs 45.11, only 1.13x.**

**And the clean decomposition: 2.49x = 1.48x (more kernel work) x 1.68x (B200's
concurrency).** MI355X issues 48 % more kernel-time and cannot overlap any of
it, which is what turns 48 % into 149 %.

So the ordering question — "is it `moe` or `attn`" — has different answers by
construction, and both are right: on elapsed time `moe` dominates (+27.67), on
per-call kernel work the buckets are close (gemm+moe 1.13x) and the outliers are
the two named kernels. **Neither table alone characterises the gap.**

### Caveat retained on the earlier "gemm absorbs wait" claim

§2c inferred that `gemm_1d1d_impl` absorbs group wait from its being 22.48 µs*
at bs=1 and 19.36 at bs=12 on an unchanged 396 calls. That inference is **not
safe**: at bs=1 `moe` is also larger (16.80 vs 13.70), so the same data is
equally explained by more overlap with a longer MoE kernel, i.e. by the ~5 %
contention measured here. The `mega_moe_impl` absorption claim is unaffected.
(*ms, not µs — 22.48 ms summed over 396 calls.)

---

## Identifying B200's 217 us per-layer GEMM: not routed MoE, most likely the
## alt-stream shared expert waiting on the a2a
*(b200, 2026-09-16. Withdraws an unverified claim of its own.)*

An earlier note called this kernel "a per-layer dense GEMM deliberately
overlapped with the MoE all-to-all, probably the shared expert". **That was
speculation.** Here is what is now measured.

### Method: attribute in EXTEND, because decode cannot be attributed at all

Decode steps are cuda-graph replays — `graph id` 32/92 on every kernel, and
**0 of 15,732** `gemm_1d1d_impl` calls resolve to a CPU op, because the launches
happened at capture time outside the window. **EXTEND steps are eager**
(`graph id` 0) and **670 of 792** resolve. Use the GPU kernel event's own
`External id` against `cpu_op` — the `cuda_runtime` correlation path only
intersects 76 of 948 ids and is the wrong route.

This works on the existing captures; no re-profiling was needed.

### What the dense GEMM family actually is

All of it is `sglang::deep_gemm_fp8_fp8_bf16_nt`. Five dim sets, from the two
EXTEND steps at 6144 tokens (`[A, A_scale, B, B_scale, out]`, so N and K are
readable):

| dims | projection | calls |
|---|---|---:|
| `[6144,7168] … [2048,7168] → [6144,2048]` | fused qkv_a (q_lora 1536 + kv_lora 512) | 122 |
| `[6144,1536] … [65536,1536] → [6144,65536]` | q_b_proj | 122 |
| `[6144,16384] … [7168,16384] → [6144,7168]` | o_proj | 122 |
| `[6144,7168] … [6144,7168] → [6144,6144]` | **shared-expert gate_up** (7168→2×3072) | 122 |
| `[6144,3072] … [7168,3072] → [6144,7168]` | **shared-expert down** (3072→7168) | 122 |

So **`num_fused_shared_experts == 0` on this arm** and the alt-stream
shared-expert path at `deepseek_v2.py:1282-1287` is live — the shared expert is
its own pair of dense GEMMs, not folded into the routed experts.

### The answer, and the residual uncertainty

- **It is NOT the routed-MoE GEMM.** Routed MoE is the `GemmType=4` grouped
  instantiation (template carries `1024u, 4096u`, 16 groups). In decode that is
  61 calls/step at **18.5 us** with **0 % overlap** with the `moe` bucket. It is
  also the one thing EXTEND cannot resolve, because the MoE custom ops register
  no shapes — the same gap MI355X reported for `record_shapes`.
- **The 217 us kernel is `GemmType=0`, i.e. dense**, and the shared expert is
  one of the five dense candidates above.
- **The exact projection cannot be pinned**, because the tile instantiation is
  chosen from M, and M is 6144 in EXTEND against ~40 in decode. Instantiations
  do not carry across the two windows.

**Quantitative argument that it is not doing its own work:** at fp4 (0.5 B per
param) and TP8, the largest per-layer projection is o_proj at 117 M params, i.e.
58.7 MB dense and ~7 MB per rank — single-digit microseconds at B200's HBM rate.
The smallest observed decode group is 15 us, already several times that bound,
which is normal small-M GEMM inefficiency. **217 us is ~30x the bound for any of
these projections.** No projection's own work explains it.

Combined with the 99 % overlap against the `moe` bucket, and with the fact that
the shared expert is exactly what the code places on `alt_stream` to overlap the
MoE call, the consistent reading is: **the 217 us is the alt-stream
shared-expert GEMM, mostly waiting on the MoE all-to-all rather than computing.**
That also explains why the *same* instantiation runs 91 calls/step at 24.8 us
with 0 % overlap.

**To close it, one clean experiment:** re-capture decode with `with_stack=true`,
or compare against the breakable-cuda-graph path, where
`deepseek_v2.py:1289`'s own comment says the shared experts "overlap nothing".

**Standing lesson for both nodes:** kernel identity cannot be established inside
a graph-replayed window. Attribute in EXTEND, then carry the *symbol family*
(dense vs grouped) rather than the tile instantiation across to decode.
